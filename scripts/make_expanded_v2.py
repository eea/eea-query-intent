"""Build the English + 10-language QA'd-translation training/calibration splits.

The English-only split (data/english_only) is the base. We add the GPT-5.6-Sol
QA'd per-row translations for the 10 safety-failing languages
(data/translations/<lang>.jsonl). Each translation row is reassigned to the
same split (train/calibration) as its English source row, keyed by
template_id, so translations of calibration rows do not leak into training.

Output:
  data/expanded_v2/train.jsonl
  data/expanded_v2/calibration.jsonl

The held-out TEST split (data/multilingual/v1/test.jsonl) is untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

from eea_query_intent.languages import SUPPORTED_LANGUAGE_CODES

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "expanded_v2"

BASE = (
    ROOT / "data" / "english_only" / "train.jsonl",
    ROOT / "data" / "english_only" / "calibration.jsonl",
)
BANK = ROOT / "data" / "english" / "expanded_v1.jsonl"
TRANSLATIONS = ROOT / "data" / "translations"
# Every supported language except the English anchor that has a QA'd file.
LANGS = [
    lang
    for lang in sorted(SUPPORTED_LANGUAGE_CODES)
    if lang != "en" and (TRANSLATIONS / f"{lang}.jsonl").exists()
]


def load(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main() -> int:
    # template_id -> split from the English bank
    split_of: dict[str, str] = {r["template_id"]: r["split"] for r in load(BANK)}

    by_split: dict[str, list[dict]] = {"train": [], "calibration": []}
    seen: set[tuple[str, str]] = set()

    def add(row: dict) -> None:
        key = (row["language"], " ".join(row["text"].split()).lower())
        if key in seen:
            return
        seen.add(key)
        by_split[row["split"]].append(row)

    for path in BASE:
        for row in load(path):
            add(row)

    for lang in LANGS:
        tpath = TRANSLATIONS / f"{lang}.jsonl"
        for row in load(tpath):
            row = dict(row)
            row["split"] = split_of.get(row["template_id"], "train")
            add(row)

    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "calibration"):
        out = OUT / f"{split}.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for row in by_split[split]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        langs: dict[str, int] = {}
        for row in by_split[split]:
            langs[row["language"]] = langs.get(row["language"], 0) + 1
        print(f"{out.name}: {len(by_split[split])} rows  langs={langs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
