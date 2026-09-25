"""Generate a large, diverse native-language TRAINING corpus.

Same realistic-generation approach as the acceptance set (see
docs/acceptance-spec.md), but for training: fresh rows per language,
disjoint from the acceptance holdout and every existing dataset
(fresh generations + case-insensitive dedup against all of them).

Quotas weight the distribution toward the hard boundary the model fails:
retrieval (incl. many "topic + location" phrases) vs. full-sentence
eligible queries.

Output: data/training/v1/<lang>.raw.jsonl (review_status unreviewed).
Run scripts/gpt_train_qa.py afterwards to flip to llm_reviewed.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from gpt_acceptance_gen import NAMES, build_prompt, call_gpt  # noqa: E402

# 3000 rows/language, weighted toward the hard no-AI/eligible boundary
TRAIN_QUOTAS = {
    "question": 750,
    "exploratory": 600,
    "claim": 600,
    "retrieval": 900,
    "unknown": 150,
}

OUT_DIR = ROOT / "data" / "training" / "v1"

# every dataset a training row must stay disjoint from
DEDUP_DIRS = [
    ROOT / "data" / "acceptance" / "v1",
    ROOT / "data" / "acceptance" / "v2",
    ROOT / "data" / "pilot" / "v1",
    ROOT / "data" / "banks" / "v1-short",
]


def existing_texts() -> set[str]:
    seen = set()
    dirs = DEDUP_DIRS + [OUT_DIR]
    for d in dirs:
        if not d.exists():
            continue
        for path in d.glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if isinstance(rec, dict) and "text" in rec:
                        seen.add(rec["text"].casefold())
    return seen


def generate_intent(
    lang: str, intent: str, n: int, batch: int, avoid: set[str]
) -> list[str]:
    rows: list[str] = []
    seen = set(avoid)
    stale = 0
    while len(rows) < n:
        need = min(batch, n - len(rows))
        got = call_gpt(build_prompt(lang, intent, need))
        fresh = []
        for r in got:
            key = r.casefold()
            if key not in seen:
                fresh.append(r)
                seen.add(key)
                if len(fresh) == need:
                    break
        rows.extend(fresh)
        print(f"{lang}/{intent} {len(rows)}/{n}", flush=True)
        # Escape hatch: the model can keep re-emitting phrases it already
        # produced (which this loop cannot show it), stalling the final row
        # forever. After 5 consecutive no-progress batches accept the small
        # shortfall; the QA top-up plus quota tolerance absorb it.
        if not fresh:
            stale += 1
            if stale >= 5:
                print(
                    f"{lang}/{intent}: 5 consecutive no-progress batches - "
                    f"accepting {len(rows)}/{n}",
                    flush=True,
                )
                break
        else:
            stale = 0
    return rows[:n]


def main() -> None:
    lang = sys.argv[1]
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 75
    if lang not in NAMES:
        sys.exit(f"unknown language {lang}; expected one of {sorted(NAMES)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{lang}.raw.jsonl"

    # crash-resilient resume: completed intents are skipped, the raw file is
    # rewritten after every intent so a mid-run crash loses at most one intent
    records: list[dict] = []
    if out.exists():
        records = [
            json.loads(line)
            for line in out.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    per_intent = dict.fromkeys(TRAIN_QUOTAS, 0)
    for rec in records:
        per_intent[rec["intent"]] += 1
    avoid = existing_texts()
    for rec in records:
        avoid.add(rec["text"].casefold())

    for intent, n_gen in TRAIN_QUOTAS.items():
        have = per_intent[intent]
        if have >= n_gen:
            print(
                f"{lang}/{intent} already complete ({have}/{n_gen}), skipping",
                flush=True,
            )
            continue
        texts = generate_intent(lang, intent, n_gen - have, batch, avoid)
        for seq, text in enumerate(texts, start=have + 1):
            row_id = f"trn-{lang}-{intent}-{seq:04d}"
            records.append(
                {
                    "id": row_id,
                    "template_id": row_id,
                    "language": lang,
                    "text": text,
                    "intent": intent,
                    "source_type": "synthetic_generated",
                    "review_status": "unreviewed",
                    "split": "train",
                }
            )
            avoid.add(text.casefold())
        with out.open("w", encoding="utf-8") as handle:
            for rec in records:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"{lang}/{intent} done ({have + len(texts)}/{n_gen}), file saved",
            flush=True,
        )
        time.sleep(2)
    print(f"{lang}: wrote {out} ({len(records)} rows)", flush=True)


if __name__ == "__main__":
    main()
