"""Build the noq-short candidate training mix.

Two corpus operations on top of data/expanded_v3/train.jsonl:

Fix A - question-mark strip: for the 17 in-house languages, ~75% of
question rows lose their trailing '?' (every 4th keeps it), so the model
no longer treats the '?' as a near-perfect question discriminator.
Deterministic on file order; the exam and all other classes untouched.

Fix B - short-question bank: appends the 120-row hand-authored English
bank plus its 27 MT translations (data/pilot/noq-short/<lang>.jsonl) as
question rows, so short (2-4 word) interrogatives are no longer outside
the question distribution.

Fix C - counter-examples: appends hand-authored counter-example rows
(data/pilot/noq-short/counter-<lang>.jsonl) that target observed
false-positive / recall-drop patterns from the frozen exam.

Outputs (gitignored pilot area; production models/setfit untouched):
  data/pilot/noq-short/train.jsonl        candidate training mix
  data/pilot/noq-short/calibration.jsonl  carried through unchanged

With --lowercase, the text field of every emitted row (mix and
calibration) is lowercased; the model is then trained and probed on
lowercase-only input, matching the service-side normalization that will
ship with promotion. Gold labels and ids are untouched.

Usage: uv run python scripts/noq_make_mix.py [--lowercase]
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_MIX = ROOT / "data" / "expanded_v3" / "train.jsonl"
SRC_CAL = ROOT / "data" / "expanded_v3" / "calibration.jsonl"
BANK_DIR = ROOT / "data" / "pilot" / "noq-short"
OUT_TRAIN = BANK_DIR / "train.jsonl"
OUT_CAL = BANK_DIR / "calibration.jsonl"

INHOUSE = {
    "bg",
    "da",
    "de",
    "en",
    "es",
    "fi",
    "fr",
    "hr",
    "it",
    "nb",
    "nl",
    "nn",
    "pl",
    "pt",
    "ro",
    "sk",
    "tr",
}
BANK_LANGS = [
    "en",
    "bg",
    "da",
    "de",
    "es",
    "fi",
    "fr",
    "hr",
    "it",
    "nb",
    "nl",
    "nn",
    "pl",
    "pt",
    "ro",
    "sk",
    "tr",
    "cs",
    "el",
    "et",
    "hu",
    "lt",
    "lv",
    "sl",
    "sv",
    "ga",
    "is",
    "mt",
]


def main() -> int:
    lowercase = "--lowercase" in sys.argv
    rows: list[dict] = []
    with SRC_MIX.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    print(f"base mix: {len(rows)} rows from {SRC_MIX.name}")

    # --- Fix A: strip the trailing '?' from ~75% of in-house question rows
    counter: Counter[str] = Counter()
    stripped = 0
    kept_marked = 0
    for row in rows:
        if (
            row.get("language") in INHOUSE
            and row.get("intent") == "question"
            and row["text"].endswith("?")
        ):
            counter[row["language"]] += 1
            if counter[row["language"]] % 4 == 0:
                kept_marked += 1
            else:
                row["text"] = row["text"][:-1]
                stripped += 1
    print(
        f"fix A: stripped '?' from {stripped} question rows "
        f"({kept_marked} kept, ~{kept_marked + stripped} in-house question rows total)"
    )
    per_lang = dict(sorted(counter.items()))
    print(f"fix A per-language question rows seen: {per_lang}")

    # --- Fix B: append the short-question bank (en + 27 translations)
    mix_texts = {r["text"].casefold() for r in rows}
    added: Counter[str] = Counter()
    collision = 0
    for lang in BANK_LANGS:
        path = BANK_DIR / f"{lang}.jsonl"
        if not path.exists():
            print(f"fix B: MISSING {path.name} - skipping language")
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = rec["text"].casefold()
                if key in mix_texts:
                    collision += 1
                    continue
                mix_texts.add(key)
                rows.append(rec)
                added[lang] += 1
    print(
        f"fix B: added {sum(added.values())} short-question rows "
        f"({collision} collision drops)"
    )
    print(f"fix B per-language: {dict(sorted(added.items()))}")

    # --- Fix C: append hand-authored counter-example rows
    counter_added: Counter[str] = Counter()
    counter_collision = 0
    for path in sorted(BANK_DIR.glob("counter-*.jsonl")):
        lang = path.stem.removeprefix("counter-")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = rec["text"].casefold()
                if key in mix_texts:
                    counter_collision += 1
                    continue
                mix_texts.add(key)
                rows.append(rec)
                counter_added[lang] += 1
    print(
        f"fix C: added {sum(counter_added.values())} counter-example rows "
        f"({counter_collision} collision drops)"
    )
    print(f"fix C per-language: {dict(sorted(counter_added.items()))}")

    BANK_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_TRAIN.open("w", encoding="utf-8") as handle:
        for rec in rows:
            if lowercase:
                rec["text"] = rec["text"].lower()
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if lowercase:
        cal_in = SRC_CAL.open(encoding="utf-8")
        cal_out = OUT_CAL.open("w", encoding="utf-8")
        with cal_in, cal_out:
            for line in cal_in:
                if line.strip():
                    cal = json.loads(line)
                    cal["text"] = cal["text"].lower()
                    cal_out.write(json.dumps(cal, ensure_ascii=False) + "\n")
        print("lowercase: applied to mix and calibration")
    else:
        shutil.copyfile(SRC_CAL, OUT_CAL)
    print(f"total: {len(rows)} rows -> {OUT_TRAIN}")
    print(f"calibration carried to {OUT_CAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
