"""GPT-5.6-Sol adversarial QA pass over a language's raw acceptance rows.

Reads ``data/acceptance/v1/<lang>.raw.jsonl`` (unreviewed, buffered counts),
asks GPT-5.6 Sol to review every row (label correctness, native grammar,
search-box register, duplicates), applies ok/fix/drop verdicts, tops up any
intent that falls below its target, prunes to the target counts, renumbers
ids, and writes the final ``data/acceptance/v1/<lang>.jsonl`` with
``review_status: llm_reviewed``. The full verdict log goes to
``reports/acceptance_qa_<lang>.json``.

Usage: uv run python scripts/gpt_acceptance_qa.py <lang>
"""

import json
import sys
from pathlib import Path

from gpt_acceptance_gen import (
    NAMES,
    QA_MODEL,
    build_prompt,
    call_gpt,
    call_gpt_raw,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "acceptance" / "v1"
REPORT_DIR = ROOT / "reports"

TARGETS = {
    "question": 150,
    "exploratory": 120,
    "claim": 90,
    "retrieval": 350,
    "unknown": 50,
}

QA_TMPL = """You are an adversarial native @NAME@ reviewer and a data-quality auditor. Below are numbered search-box queries for the European Environment Agency website, each with an intent label. For every row return exactly one verdict:
- "ok": the text is grammatical and natural for a native speaker, reads like a real search-box entry, and matches its label's definition.
- "fix": the intent is right but the text is ungrammatical, unnatural, not in native @NAME@, or borderline violates its definition in a fixable way. Provide the corrected text in "t".
- "drop": unusable (intent wrong beyond repair, broken or empty text, or a duplicate of another row in this batch).

Label definitions:
- question: @DEF_QUESTION@
- exploratory: @DEF_EXPLORATORY@
- claim: @DEF_CLAIM@
- retrieval: @DEF_RETRIEVAL@
- unknown: @DEF_UNKNOWN@

Extra checks: text must be in @NAME@ (Latin acronyms, proper nouns, digits are fine), at most 20 words, plausible as something a real visitor would type.

Output STRICT JSON only, no prose, no markdown: {"rows":[{"i":1,"v":"ok","t":"","r":"why"}]} - one entry per numbered row, in order. "t" is filled only for "fix"; "r" is at most 8 words.

Rows:
"""


def qa_batch(rows: list[tuple[int, str, str]], lang: str) -> dict:
    from gpt_acceptance_gen import DEFINITIONS

    prompt = (
        QA_TMPL.replace("@NAME@", NAMES[lang])
        .replace("@DEF_QUESTION@", DEFINITIONS["question"])
        .replace("@DEF_EXPLORATORY@", DEFINITIONS["exploratory"])
        .replace("@DEF_CLAIM@", DEFINITIONS["claim"])
        .replace("@DEF_RETRIEVAL@", DEFINITIONS["retrieval"])
        .replace("@DEF_UNKNOWN@", DEFINITIONS["unknown"])
    )
    for i, _intent, text in rows:
        prompt += f"{i} [{_intent}] {text}\n"
    return call_gpt_raw(prompt, model=QA_MODEL)


def main() -> None:
    lang = sys.argv[1]
    raw_path = DATA_DIR / f"{lang}.raw.jsonl"
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    verdicts: dict[int, dict] = {}
    for start in range(0, len(rows), 200):
        chunk = rows[start : start + 200]
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
            f"{lang} QA batch {start + 1}-{start + len(chunk)}/{len(rows)}", flush=True
        )

    # apply verdicts, grouped by intent in original order
    kept: dict[str, list[str]] = {intent: [] for intent in TARGETS}
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

    # top-up any intent below target (max 2 cycles)
    topups = 0
    for _cycle in range(2):
        shortfalls = {
            intent: TARGETS[intent] - len(texts) for intent, texts in kept.items()
        }
        need = {i: n for i, n in shortfalls.items() if n > 0}
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

    # final safety: dedupe (case-insensitive), cap word count, prune to target
    final: dict[str, list[str]] = {}
    for intent, texts in kept.items():
        seen = set()
        clean = []
        for t in texts:
            key = t.casefold()
            if not t or len(t.split()) > 20 or key in seen:
                continue
            seen.add(key)
            clean.append(t)
        final[intent] = clean[: TARGETS[intent]]

    counts = {intent: len(texts) for intent, texts in final.items()}
    log = {
        "language": lang,
        "raw_rows": len(rows),
        "verdicts": stats,
        "topups": topups,
        "final_counts": counts,
        "complete": all(counts[i] == TARGETS[i] for i in TARGETS),
    }
    if not log["complete"]:
        # Do not write a truncated shard: a partial output would look like a
        # finished acceptance set to the pipeline.
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / f"acceptance_qa_{lang}.json").write_text(
            json.dumps(log, indent=2), encoding="utf-8"
        )
        print(f"{lang}: below target after QA: {counts}", flush=True)
        sys.exit(f"{lang}: below target after QA: {counts}")

    out_path = DATA_DIR / f"{lang}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for intent in ("question", "exploratory", "claim", "retrieval", "unknown"):
            for seq, text in enumerate(final[intent], start=1):
                row_id = f"acc-{lang}-{intent}-{seq:04d}"
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
                            "split": "test",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"acceptance_qa_{lang}.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )
    print(f"{lang}: wrote {out_path} counts={counts}", flush=True)


if __name__ == "__main__":
    main()
