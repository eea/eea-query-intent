"""Build the English validation training set.

Merges the frozen English base (data/english_only/train.jsonl) with the new
diverse English training corpus (data/training/v1/en.jsonl), deduplicated
case-insensitively, into data/en_validation/train.jsonl. Also copies the
English calibration split through so the validation model keeps the same
calibration input as the English-only baseline.

Usage: uv run python scripts/make_en_validation.py
"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EN_BASE = ROOT / "data" / "english_only" / "train.jsonl"
EN_CORPUS = ROOT / "data" / "training" / "v1" / "en.jsonl"
OUT_DIR = ROOT / "data" / "en_validation"
OUT_TRAIN = OUT_DIR / "train.jsonl"
OUT_CAL = OUT_DIR / "calibration.jsonl"


def load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    base = load(EN_BASE)
    corpus = load(EN_CORPUS)
    seen = {r["text"].casefold() for r in base}
    merged = list(base)
    added = 0
    for rec in corpus:
        key = rec["text"].casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(rec)
        added += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_TRAIN.open("w", encoding="utf-8") as handle:
        for rec in merged:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    shutil.copyfile(ROOT / "data" / "english_only" / "calibration.jsonl", OUT_CAL)

    by_intent: dict[str, int] = {}
    for rec in merged:
        by_intent[rec["intent"]] = by_intent.get(rec["intent"], 0) + 1
    print(f"base={len(base)} corpus={len(corpus)} added={added} total={len(merged)}")
    print(f"by intent: {by_intent}")
    print(f"wrote {OUT_TRAIN} and {OUT_CAL}")


if __name__ == "__main__":
    main()
