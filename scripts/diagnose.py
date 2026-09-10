"""Diagnose routing errors for a candidate model against the test split."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ELIGIBLE = {"question", "exploratory", "claim"}


def main() -> int:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models" / "fasttext"
    gold = {
        json.loads(line)["id"]: json.loads(line)
        for line in (ROOT / "data/multilingual/v1/test.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    preds = {
        json.loads(line)["id"]: json.loads(line)
        for line in (model_dir / "test-predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }

    fp = [
        (g, preds[g["id"]])
        for g in gold.values()
        if g["intent"] not in ELIGIBLE and preds[g["id"]]["eligible"]
    ]
    fn = [
        (g, preds[g["id"]])
        for g in gold.values()
        if g["intent"] in ELIGIBLE and not preds[g["id"]]["eligible"]
    ]
    print(f"false AI routes (FP): {len(fp)}   missed eligible (FN): {len(fn)}")
    print("\nFPs by language:")
    by_lang: dict[str, int] = {}
    for g, _ in fp:
        by_lang[g["language"]] = by_lang.get(g["language"], 0) + 1
    for lang, count in sorted(by_lang.items(), key=lambda kv: -kv[1]):
        print(f"  {lang}: {count}")
    print("\nsample FPs:")
    for g, p in fp[:12]:
        print(
            f"  {g['language']} GOLD {g['intent']:<12s} -> "
            f"{p['intent']:<12s} p={p['eligible_probability']:.2f}  {g['text']!r}"
        )
    print("\nsample FNs:")
    for g, p in fn[:8]:
        print(
            f"  {g['language']} GOLD {g['intent']:<12s} -> "
            f"{p['intent']:<12s} p={p['eligible_probability']:.2f}  {g['text']!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
