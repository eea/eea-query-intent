"""Build the final all-language training set (expanded_v3).

Folds the per-language diverse training corpus (data/training/v1/<lang>.jsonl,
llm_reviewed) into the expanded_v2 base, deduplicated case-insensitively,
into data/expanded_v3/train.jsonl. The calibration split is carried over
from expanded_v2 unchanged.

Usage: uv run python scripts/make_final_v3.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASE_TRAIN = ROOT / "data" / "expanded_v2" / "train.jsonl"
BASE_CAL = ROOT / "data" / "expanded_v2" / "calibration.jsonl"
CORPUS_DIR = ROOT / "data" / "training" / "v1"
OUT_DIR = ROOT / "data" / "expanded_v3"


def load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    base = load(BASE_TRAIN)
    seen = {r["text"].casefold() for r in base}
    merged = list(base)
    per_lang: dict[str, int] = {}
    for path in sorted(CORPUS_DIR.glob("*.jsonl")):
        lang = path.stem
        if lang == "raw" or path.name.endswith(".raw.jsonl"):
            continue
        added = 0
        for rec in load(path):
            key = rec["text"].casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(rec)
            added += 1
        per_lang[lang] = added

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "train.jsonl").open("w", encoding="utf-8") as handle:
        for rec in merged:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (OUT_DIR / "calibration.jsonl").open("w", encoding="utf-8") as handle:
        for rec in load(BASE_CAL):
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"base={len(base)} total={len(merged)}")
    print(f"added per language: {per_lang}")
    print(f"wrote {OUT_DIR}/train.jsonl and {OUT_DIR}/calibration.jsonl")


if __name__ == "__main__":
    main()
