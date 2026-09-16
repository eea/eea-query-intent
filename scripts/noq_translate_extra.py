"""Second translation pass: translate the extra short exploratory/claim
rows (data/pilot/noq-short/en_extra.jsonl) and APPEND them to the
existing per-language bank files from the main pass.

Must run only after the main pass (noq_translate_short.py) has finished,
so it never fights a concurrent writer over the bank files.

Usage: uv run --with sentencepiece python scripts/noq_translate_extra.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from noq_translate_short import (
    ALL_LANGS,
    NLLB_TARGETS,
    NllbEngine,
    OpusEngine,
    blocked_texts,
    translate_language,
)

ROOT = Path(__file__).resolve().parent.parent
EN_EXTRA = ROOT / "data" / "pilot" / "noq-short" / "en_extra.jsonl"


def load_extra() -> list[dict]:
    return [
        json.loads(line)
        for line in EN_EXTRA.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    langs = sorted(ALL_LANGS)
    rows = load_extra()
    print(f"source: {EN_EXTRA} ({len(rows)} rows)")
    blocked = blocked_texts()

    nllb_langs = [code for code in langs if code in NLLB_TARGETS]
    opus_langs = [code for code in langs if code in ("mt",)]

    if nllb_langs:
        engine = NllbEngine()
        print(f"targets verified: {len(NLLB_TARGETS)} NLLB codes", flush=True)
        for lang in nllb_langs:
            t0 = time.time()
            translate_language(
                lang,
                rows,
                blocked,
                lambda texts, _t=NLLB_TARGETS[lang]: engine.translate(_t, texts),
                append=True,
            )
            print(f"[{lang}] extra done in {(time.time() - t0) / 60:.1f}m", flush=True)
        del engine

    if opus_langs:
        engine = OpusEngine()
        for lang in opus_langs:
            t0 = time.time()
            translate_language(
                lang,
                rows,
                blocked,
                lambda texts: engine.translate("mt", texts),
                append=True,
            )
            print(f"[{lang}] extra done in {(time.time() - t0) / 60:.1f}m", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
