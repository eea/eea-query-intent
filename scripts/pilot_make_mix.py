"""Build the NLLB pilot training mix.

Takes data/expanded_v3/train.jsonl, removes the 413-row GPT-era leftovers
for the pilot languages, and adds the NLLB-translated corpora
(data/pilot/nllb/<lang>.jsonl) in their place, case-insensitively
deduplicated. Calibration is carried over unchanged.

Usage: uv run python scripts/pilot_make_mix.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_TRAIN = ROOT / "data" / "expanded_v3" / "train.jsonl"
SRC_CAL = ROOT / "data" / "expanded_v3" / "calibration.jsonl"
PILOT_DIR = ROOT / "data" / "pilot" / "nllb"
OUT_TRAIN = PILOT_DIR / "train.jsonl"
OUT_CAL = PILOT_DIR / "calibration.jsonl"

PILOT_LANGS = ("mt", "ga", "is")


def load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    base = load(SRC_TRAIN)
    removed = 0
    kept: list[dict] = []
    seen: set[str] = set()
    for rec in base:
        if rec.get("language") in PILOT_LANGS:
            removed += 1
            continue
        kept.append(rec)
        seen.add(rec["text"].casefold())

    added = {lang: 0 for lang in PILOT_LANGS}
    for lang in PILOT_LANGS:
        path = PILOT_DIR / f"{lang}.jsonl"
        for rec in load(path):
            key = rec["text"].casefold()
            if key in seen:
                continue
            seen.add(key)
            kept.append(rec)
            added[lang] += 1

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_TRAIN.open("w", encoding="utf-8") as handle:
        for rec in kept:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with OUT_CAL.open("w", encoding="utf-8") as handle:
        for rec in load(SRC_CAL):
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"base={len(base)} removed_legacy={removed} added={added}")
    print(f"total={len(kept)} -> {OUT_TRAIN}")


if __name__ == "__main__":
    main()
