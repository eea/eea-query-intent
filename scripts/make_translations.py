"""Build reviewable per-language translation JSONL from the English bank.

For each target language, ``scripts/translations/<lang>.py`` defines a
``TRANSLATIONS`` list of exactly ``len(english bank)`` strings, in the same
order as ``data/english/expanded_v1.jsonl``. This script zips the bank with
the per-language list and writes ``data/translations/<lang>.jsonl`` where
each row carries the English source next to the translation so a native
reviewer can check them side by side.

Rows are marked ``source_type=synthetic_translated``,
``review_status=unreviewed`` (a native reviewer must confirm each), and
``split=train`` (they become training data once reviewed). A length
mismatch between a language list and the bank fails hard.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

BANK = ROOT / "data" / "english" / "expanded_v1.jsonl"
OUT = ROOT / "data" / "translations"
LANGS = ["et", "fi", "is", "it", "lt", "lv", "nl", "pt", "sv", "tr"]


def main() -> int:
    bank = [
        json.loads(line)
        for line in BANK.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    missing = []
    for lang in LANGS:
        try:
            module = importlib.import_module(f"translations.{lang}")
        except ModuleNotFoundError:
            missing.append(lang)
            continue
        translations = module.TRANSLATIONS
        if len(translations) != len(bank):
            raise SystemExit(
                f"{lang}: {len(translations)} translations != {len(bank)} bank rows"
            )
        out = OUT / f"{lang}.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for row, text in zip(bank, translations, strict=True):
                handle.write(
                    json.dumps(
                        {
                            "id": f"{lang}-{row['template_id']}",
                            "template_id": row["template_id"],
                            "language": lang,
                            "english": row["text"],
                            "text": text,
                            "intent": row["intent"],
                            "source_type": "synthetic_translated",
                            "review_status": "unreviewed",
                            "split": "train",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"{lang}: wrote {out} ({len(bank)} rows)")
    if missing:
        print(f"skipped (no module yet): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
