"""Assemble the English-only training and calibration splits.

Sources (deduplicated case-insensitively by text, first source wins):
  1. data/seed/english.jsonl                 (118 frontend corpus rows, train)
  2. data/multilingual/v1/train.jsonl        (en template rows, train)
  3. data/multilingual/v1/calibration.jsonl  (en template rows, calibration)
  4. data/english/expanded_v1.jsonl          (authored bank, 90/10 split)

Output:
  data/english_only/train.jsonl
  data/english_only/calibration.jsonl

The held-out TEST split for per-language evaluation stays in
data/multilingual/v1/test.jsonl (all 28 languages) and is not modified.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "english_only"

SOURCES = (
    ROOT / "data" / "seed" / "english.jsonl",
    ROOT / "data" / "multilingual" / "v1" / "train.jsonl",
    ROOT / "data" / "multilingual" / "v1" / "calibration.jsonl",
    ROOT / "data" / "english" / "expanded_v1.jsonl",
)


def load(path: Path, en_only: bool) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if en_only and row["language"] != "en":
            continue
        rows.append(row)
    return rows


def main() -> int:
    seen: set[str] = set()
    by_split: dict[str, list[dict]] = {"train": [], "calibration": []}
    for path in SOURCES:
        en_only = path.name in ("train.jsonl", "calibration.jsonl")
        for row in load(path, en_only=en_only):
            key = " ".join(row["text"].split()).lower()
            if key in seen:
                continue
            seen.add(key)
            by_split[row["split"]].append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "calibration"):
        out = OUT / f"{split}.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for row in by_split[split]:
                handle.write(json.dumps(row) + "\n")
        intents: dict[str, int] = {}
        for row in by_split[split]:
            intents[row["intent"]] = intents.get(row["intent"], 0) + 1
        print(f"{out.name}: {len(by_split[split])} rows  {intents}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
