"""GPT-5.6-Sol QA pass over a language's raw TRAINING rows.

Mirror of the acceptance QA (scripts/gpt_acceptance_qa.py) for
data/training/v1: applies ok/fix/drop verdicts, tops up intents below
TRAIN_QUOTAS, prunes, and writes data/training/v1/<lang>.jsonl with
review_status llm_reviewed and split train. Final rows are re-checked
against every other dataset so a QA "fix" can never leak an acceptance
holdout text into training.

Usage: uv run python scripts/gpt_train_qa.py <lang>
"""

import json
import sys
from pathlib import Path

from gpt_acceptance_gen import build_prompt, call_gpt
from gpt_acceptance_qa import qa_batch
from gpt_train_gen import DEDUP_DIRS, OUT_DIR, TRAIN_QUOTAS

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"


def final_avoid_set(exclude: Path) -> set[str]:
    seen = set()
    dirs = [d for d in DEDUP_DIRS + [OUT_DIR] if d.exists()]
    for d in dirs:
        for path in d.glob("*.jsonl"):
            if path == exclude:
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    seen.add(json.loads(line)["text"].casefold())
    return seen


def main() -> None:
    lang = sys.argv[1]
    raw_path = OUT_DIR / f"{lang}.raw.jsonl"
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    verdicts: dict[int, dict] = {}
    for start in range(0, len(rows), 100):
        chunk = rows[start : start + 100]
        numbered = [
            (start + j, rec["intent"], rec["text"]) for j, rec in enumerate(chunk)
        ]
        res = qa_batch(numbered, lang)
        by_i = {r["i"]: r for r in res.get("rows", [])}
        for i, _intent, _text in numbered:
            v = by_i.get(i)
            verdicts[i] = (
                v
                if v is not None
                else {"v": "drop", "t": "", "r": "missing-from-output"}
            )
        print(
            f"{lang} train-QA batch {start + 1}-{start + len(chunk)}/{len(rows)}",
            flush=True,
        )

    kept: dict[str, list[str]] = {intent: [] for intent in TRAIN_QUOTAS}
    stats = {"ok": 0, "fix": 0, "drop": 0}
    for idx, rec in enumerate(rows):
        v = verdicts[idx]
        kind = v.get("v", "drop")
        if kind == "ok":
            kept[rec["intent"]].append(rec["text"].strip())
            stats["ok"] += 1
        elif kind == "fix":
            corrected = (v.get("t") or "").strip()
            if corrected:
                kept[rec["intent"]].append(corrected)
                stats["fix"] += 1
            else:
                stats["drop"] += 1
        else:
            stats["drop"] += 1

    topups = 0
    for _cycle in range(2):
        need = {
            intent: TRAIN_QUOTAS[intent] - len(texts)
            for intent, texts in kept.items()
            if len(texts) < TRAIN_QUOTAS[intent]
        }
        if not need:
            break
        for intent, n in need.items():
            generated = call_gpt(build_prompt(lang, intent, n))
            res = qa_batch(
                [(j, intent, text) for j, text in enumerate(generated, start=1)], lang
            )
            by_i = {r["i"]: r for r in res.get("rows", [])}
            seen = {t.casefold() for t in kept[intent]}
            for j, text in enumerate(generated, start=1):
                v = by_i.get(j)
                if v and v.get("v") in ("ok", "fix"):
                    t = (v.get("t") or text).strip()
                    if t and t.casefold() not in seen:
                        kept[intent].append(t)
                        seen.add(t.casefold())
                        topups += 1
        print(
            f"{lang} after top-up: "
            + ", ".join(f"{i}={len(t)}" for i, t in kept.items()),
            flush=True,
        )

    # final safety: dedupe, cap word count, drop anything that collides with
    # another dataset (incl. the acceptance holdout), prune to target
    avoid = final_avoid_set(OUT_DIR / f"{lang}.jsonl")
    final: dict[str, list[str]] = {}
    leaked = 0
    for intent, texts in kept.items():
        seen = set()
        clean = []
        for t in texts:
            key = t.casefold()
            if not t or len(t.split()) > 20 or key in seen or key in avoid:
                leaked += int(key in avoid)
                continue
            seen.add(key)
            clean.append(t)
        final[intent] = clean[: TRAIN_QUOTAS[intent]]

    out_path = OUT_DIR / f"{lang}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for intent in ("question", "exploratory", "claim", "retrieval", "unknown"):
            for seq, text in enumerate(final[intent], start=1):
                row_id = f"trn-{lang}-{intent}-{seq:04d}"
                handle.write(
                    json.dumps(
                        {
                            "id": row_id,
                            "template_id": row_id,
                            "language": lang,
                            "text": text,
                            "intent": intent,
                            "source_type": "synthetic_generated",
                            "review_status": "llm_reviewed",
                            "split": "train",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    counts = {intent: len(texts) for intent, texts in final.items()}
    log = {
        "language": lang,
        "raw_rows": len(rows),
        "verdicts": stats,
        "topups": topups,
        "leaked_dropped": leaked,
        "final_counts": counts,
        "complete": all(counts[i] == TRAIN_QUOTAS[i] for i in TRAIN_QUOTAS),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"train_qa_{lang}.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )
    print(f"{lang}: wrote {out_path} counts={counts}", flush=True)
    if not log["complete"]:
        sys.exit(f"{lang}: below target after QA: {counts}")


if __name__ == "__main__":
    main()
