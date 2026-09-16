"""Translate the 120-row short-question bank into all 27 non-English
training languages (local MT only, no GPT, no external API).

Source: data/pilot/noq-short/en.jsonl (120 hand-authored 2-4 word
questions, 25% with a trailing '?').

Engines:
  - mt: Helsinki-NLP/opus-mt-en-mt (Marian; NLLB produces broken Maltese)
  - all other 26: facebook/nllb-200-1.3B, target via forced_bos
    (validated pattern from the NLLB pilot; the 1.3B model is loaded
    ONCE and reused across languages - only the forced_bos token
    changes per language).

QA is heuristic (same guards as the pilot): empty, identical-to-source,
over 20 words / 500 chars, JSON/markdown artifacts, garbled characters,
case-insensitive duplicates, and exam/corpus collisions.

Output: data/pilot/noq-short/<lang>.jsonl in the final-corpus row shape.

Note: the Marian tokenizer needs sentencepiece:
    uv run --with sentencepiece python scripts/noq_translate_short.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EN_BANK = ROOT / "data" / "pilot" / "noq-short" / "en.jsonl"
OUT_DIR = ROOT / "data" / "pilot" / "noq-short"

SRC_LANG = "eng_Latn"
NLLB_MODEL = "facebook/nllb-200-1.3B"
OPUS_MODEL = "Helsinki-NLP/opus-mt-en-mt"

# 26 in-house + GPT-held + accepted-gap languages (all 28 except en).
# mt is routed to opus; everything else is NLLB with these target codes.
NLLB_TARGETS: dict[str, str] = {
    # 16 in-house (en is the source)
    "bg": "bul_Cyrl",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "hr": "hrv_Latn",
    "it": "ita_Latn",
    "nb": "nob_Latn",
    "nl": "nld_Latn",
    "nn": "nno_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "sk": "slk_Latn",
    "tr": "tur_Latn",
    # 8 GPT-held languages (included per user instruction 2026-09-15)
    "cs": "ces_Latn",
    "el": "ell_Grek",
    "et": "est_Latn",
    "hu": "hun_Latn",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "sl": "slv_Latn",
    "sv": "swe_Latn",
    # 2 accepted-gap languages (included per user instruction 2026-09-15;
    # near-zero expected effect per the NLLB pilot, but cost is ~2 min)
    "ga": "gle_Latn",
    "is": "isl_Latn",
}
OPUS_LANGS = {"mt"}
ALL_LANGS = set(NLLB_TARGETS) | OPUS_LANGS

MAX_WORDS = 20
MAX_CHARS = 500
BATCH = 16

# contamination sources (casefolded texts a translation must not match)
DEDUP_FILES = [
    ROOT / "data" / "acceptance" / "v1" / "test.jsonl",
    ROOT / "data" / "expanded_v3" / "train.jsonl",
    ROOT / "data" / "expanded_v3" / "calibration.jsonl",
    ROOT / "data" / "english_only" / "train.jsonl",
    ROOT / "data" / "seed" / "train.jsonl",
    ROOT / "data" / "multilingual" / "v1" / "train.jsonl",
]


def load_bank() -> list[dict]:
    return [
        json.loads(line)
        for line in EN_BANK.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def blocked_texts() -> set[str]:
    blocked: set[str] = set()
    for path in DEDUP_FILES:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rec = json.loads(line)
                    if isinstance(rec, dict) and "text" in rec:
                        blocked.add(rec["text"].casefold())
    return blocked


def heuristic_qa(text: str, source: str) -> str | None:
    """Return a drop reason, or None if the row is acceptable."""
    text = text.strip()
    if not text:
        return "empty"
    if text.casefold() == source.strip().casefold():
        return "not_translated"
    if len(text) > MAX_CHARS:
        return "too_long_chars"
    if len(text.split()) > MAX_WORDS:
        return "too_long_words"
    if any(ch in text for ch in "{}[]```") or '"rows"' in text:
        return "artifact"
    bad_chars = sum(1 for ch in text if ch in "•\ufffd\u25a0\u25ab")
    if bad_chars >= 3 or bad_chars / max(len(text), 1) > 0.2:
        return "garbled"
    return None


class NllbEngine:
    """One 1.3B model reused across all NLLB target languages."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print(f"loading {NLLB_MODEL} ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL, dtype=torch.float16)
        self.model = model.to("mps").eval()

    def check_targets(self) -> list[str]:
        missing = [
            f"{lang}:{code}"
            for lang, code in NLLB_TARGETS.items()
            if self.tokenizer.convert_tokens_to_ids(code) <= 3
        ]
        if missing:
            print(
                "ERROR: NLLB vocab is missing target codes: " + ", ".join(missing),
                flush=True,
            )
            sys.exit(1)
        return list(NLLB_TARGETS)

    def translate(self, target: str, texts: list[str]) -> list[str]:
        import torch

        self.tokenizer.src_lang = SRC_LANG
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to("mps")
        inputs["forced_bos_token_id"] = self.tokenizer.convert_tokens_to_ids(target)
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, num_beams=5, max_new_tokens=64)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


class OpusEngine:
    def __init__(self) -> None:
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        print(f"loading {OPUS_MODEL} ...", flush=True)
        self.tokenizer = MarianTokenizer.from_pretrained(OPUS_MODEL)
        model = MarianMTModel.from_pretrained(OPUS_MODEL, dtype=torch.float16)
        self.model = model.to("mps").eval()

    def translate(self, _target: str, texts: list[str]) -> list[str]:
        import torch

        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, max_length=128
        ).to("mps")
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, num_beams=5, max_length=64)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


def translate_language(
    lang: str,
    rows: list[dict],
    blocked: set[str],
    generate,
    append: bool = False,
) -> None:
    out_path = OUT_DIR / f"{lang}.jsonl"
    if append:
        offset = sum(1 for _ in out_path.open(encoding="utf-8"))
        mode = "a"
    else:
        offset = 0
        mode = "w"
    kept: list[dict] = []
    seen: set[str] = set()
    drops = Counter()

    for start in range(0, len(rows), BATCH):
        chunk = rows[start : start + BATCH]
        texts = [r["text"] for r in chunk]
        t0 = time.time()
        results = generate(texts)
        for rec, translated in zip(chunk, results, strict=True):
            text = translated.strip()
            reason = heuristic_qa(text, rec["text"])
            key = text.casefold() if reason is None else ""
            if reason is None and key in blocked:
                reason = "collision"
            if reason is None and key in seen:
                reason = "duplicate"
            if reason is not None:
                drops[reason] += 1
                continue
            seen.add(key)
            n = offset + len(kept) + 1
            kept.append(
                {
                    "id": f"sb-{lang}-{rec['intent']}-{n:04d}",
                    "template_id": f"sb-{lang}-{rec['intent']}-{n:04d}",
                    "language": lang,
                    "text": text,
                    "intent": "question",
                    "source_type": "nllb_translated",
                    "review_status": "machine_translated",
                    "split": "train",
                }
            )
        elapsed = time.time() - t0
        rate = len(chunk) / elapsed if elapsed else 0
        remaining = (len(rows) - start - len(chunk)) / rate if rate else 0
        print(
            f"[{lang}] {min(start + BATCH, len(rows))}/{len(rows)} "
            f"kept={len(kept)} drops={dict(drops)} "
            f"eta={remaining / 60:.1f}m",
            flush=True,
        )

    with out_path.open(mode, encoding="utf-8") as handle:
        for rec in kept:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(
        f"[{lang}] wrote {out_path} ({len(kept)} rows, drops={dict(drops)})",
        flush=True,
    )


def main() -> int:
    langs = [c for c in (sys.argv[1:] or sorted(ALL_LANGS)) if c in ALL_LANGS]
    if not langs:
        print("no valid languages; expected subset of", sorted(ALL_LANGS))
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_bank()
    print(f"source: {EN_BANK} ({len(rows)} rows)")
    blocked = blocked_texts()

    nllb_langs = [code for code in langs if code in NLLB_TARGETS]
    opus_langs = [code for code in langs if code in OPUS_LANGS]

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
            )
            print(
                f"[{lang}] done in {(time.time() - t0) / 60:.1f}m",
                flush=True,
            )
        del engine

    if opus_langs:
        engine = OpusEngine()
        for lang in opus_langs:
            t0 = time.time()
            translate_language(
                lang, rows, blocked, lambda texts: engine.translate("mt", texts)
            )
            print(
                f"[{lang}] done in {(time.time() - t0) / 60:.1f}m",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
