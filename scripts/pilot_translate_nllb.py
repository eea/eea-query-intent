"""Pilot: local MT translation of the English training corpus.

Replaces the 413-row GPT-era leftovers for a language with ~3000 rows
translated (not generated) from data/training/v1/en.jsonl, to test whether
machine translation can fix the weak low-resource languages without any
GPT quota or external API.

Engines (all local, no external service):
  - mt: Helsinki-NLP/opus-mt-en-mt (Marian; NLLB produces broken Maltese)
  - all others: facebook/nllb-200-1.3B (target language via forced_bos)

NLLB target codes verified empirically (2026-09-16 short-bank pass):
  bg bul_Cyrl, cs ces_Latn, da dan_Latn, de deu_Latn, el ell_Grek,
  bg bul_Cyrl, cs ces_Latn, da dan_Latn, de deu_Latn, el ell_Grek,
  es spa_Latn (NOT esp_Latn - that is <unk>), et est_Latn, fi fin_Latn,
  fr fra_Latn, ga gle_Latn, hr hrv_Latn, hu hun_Latn, is isl_Latn,
  it ita_Latn, lt lit_Latn, lv lvs_Latn, nb nob_Latn, nl nld_Latn,
  nn nno_Latn, pl pol_Latn, pt por_Latn, ro ron_Latn, sk slk_Latn,
  sl slv_Latn, sv swe_Latn, tr tur_Latn
  (bul_Latn/nor_Latn/lav_Latn do NOT exist in the 1.3B vocab)

No LLM reviewer is used (GPT is paused, and the in-house model is the
measured-weak one for exactly these languages), so QA is heuristic:
  - drop empty output
  - output identical to the English source: drop at >=4 words (a sentence
    that was not translated); keep + flag at <=3 words (acronyms and
    international terms legitimately stay unchanged)
  - drop over 20 words / over 500 chars (the service guard bounds)
  - drop JSON / markdown artifacts and undecodable garbage
  - drop case-insensitive duplicates
  - drop any row colliding with the acceptance exam or an existing corpus
    (contamination guard, mirrors gpt_train_gen.DEDUP_DIRS)

Every output row carries source_id = the English source row id (lineage).

Usage:
  uv run python scripts/pilot_translate_nllb.py sl sv --out-dir data/training/v1
  (Marian targets need: uv run --with sentencepiece python ...)
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EN_CORPUS = ROOT / "data" / "training" / "v1" / "en.jsonl"
OUT_DIR = ROOT / "data" / "pilot" / "nllb"

SRC_LANG = "eng_Latn"
NLLB_MODEL = "facebook/nllb-200-1.3B"


def _nllb(target: str) -> dict:
    return {"engine": "nllb", "model": NLLB_MODEL, "target": target}


ENGINES = {
    "mt": {"engine": "opus", "model": "Helsinki-NLP/opus-mt-en-mt", "target": "mt"},
    "bg": _nllb("bul_Cyrl"),
    "cs": _nllb("ces_Latn"),
    "da": _nllb("dan_Latn"),
    "de": _nllb("deu_Latn"),
    "el": _nllb("ell_Grek"),
    "es": _nllb("spa_Latn"),
    "et": _nllb("est_Latn"),
    "fi": _nllb("fin_Latn"),
    "fr": _nllb("fra_Latn"),
    "ga": _nllb("gle_Latn"),
    "hr": _nllb("hrv_Latn"),
    "hu": _nllb("hun_Latn"),
    "is": _nllb("isl_Latn"),
    "it": _nllb("ita_Latn"),
    "lt": _nllb("lit_Latn"),
    "lv": _nllb("lvs_Latn"),
    "nb": _nllb("nob_Latn"),
    "nl": _nllb("nld_Latn"),
    "nn": _nllb("nno_Latn"),
    "pl": _nllb("pol_Latn"),
    "pt": _nllb("por_Latn"),
    "ro": _nllb("ron_Latn"),
    "sk": _nllb("slk_Latn"),
    "sl": _nllb("slv_Latn"),
    "sv": _nllb("swe_Latn"),
    "tr": _nllb("tur_Latn"),
}

MAX_WORDS = 20
MAX_CHARS = 500

# contamination sources (casefolded texts a translation must not match)
DEDUP_FILES = [
    ROOT / "data" / "acceptance" / "v1" / "test.jsonl",
    ROOT / "data" / "acceptance" / "v2" / "test.jsonl",
    ROOT / "data" / "pilot" / "v1" / "train.jsonl",
    ROOT / "data" / "pilot" / "v1" / "calibration.jsonl",
    ROOT / "data" / "pilot" / "v1" / "calibration_old.jsonl",
]


def load_corpus(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _dedup_files() -> list[Path]:
    files = [p for p in DEDUP_FILES if p.exists()]
    bank_dir = ROOT / "data" / "banks" / "v1-short"
    if bank_dir.exists():
        for p in sorted(bank_dir.glob("*.jsonl")):
            files.append(p)
    return files


def blocked_texts() -> set[str]:
    blocked: set[str] = set()
    for path in _dedup_files():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rec = json.loads(line)
                    if isinstance(rec, dict) and "text" in rec:
                        blocked.add(rec["text"].casefold())
    return blocked


def heuristic_qa(text: str, source: str) -> tuple[str | None, str | None]:
    """Return (drop_reason, flag). drop_reason None = keep; flag is advisory."""
    text = text.strip()
    if not text:
        return "empty", None
    if text.casefold() == source.strip().casefold():
        # Acronyms / international terms legitimately stay unchanged.
        if len(text.split()) <= 3:
            return None, "identical_short"
        return "not_translated", None
    if len(text) > MAX_CHARS:
        return "too_long_chars", None
    if len(text.split()) > MAX_WORDS:
        return "too_long_words", None
    if any(ch in text for ch in "{}[]```") or '"rows"' in text:
        return "artifact", None
    # undecodable garbage (NLLB Maltese failure mode emits these)
    bad_chars = sum(1 for ch in text if ch in "•\ufffd\u25a0\u25ab")
    if bad_chars >= 3 or bad_chars / max(len(text), 1) > 0.2:
        return "garbled", None
    return None, None


def build_engine(lang: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    spec = ENGINES[lang]
    print(f"[{lang}] loading {spec['model']} ({spec['engine']}) ...", flush=True)
    if spec["engine"] == "opus":
        from transformers import MarianMTModel, MarianTokenizer

        tokenizer = MarianTokenizer.from_pretrained(spec["model"])
        model = MarianMTModel.from_pretrained(spec["model"], dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(spec["model"])
        model = AutoModelForSeq2SeqLM.from_pretrained(
            spec["model"], dtype=torch.float16
        )
    model = model.to("mps")
    model.eval()
    return tokenizer, model, spec


def generate_batch(tokenizer, model, spec: dict, texts: list[str]) -> list[str]:
    import torch

    if spec["engine"] == "nllb":
        tokenizer.src_lang = SRC_LANG
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to("mps")
        inputs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(spec["target"])
        kwargs = {"num_beams": 5, "max_new_tokens": 64}
    else:  # opus (Marian): plain seq2seq
        inputs = tokenizer(texts, return_tensors="pt", padding=True, max_length=128).to(
            "mps"
        )
        kwargs = {"num_beams": 5, "max_length": 64}
    with torch.inference_mode():
        outputs = model.generate(**inputs, **kwargs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def translate_language(
    lang: str, rows: list[dict], out_dir: Path, id_prefix: str = "nllb"
) -> None:
    tokenizer, model, spec = build_engine(lang)
    print(f"[{lang}] model loaded; translating {len(rows)} rows", flush=True)

    def row_id(rec: dict) -> str:
        parts = rec["id"].split("-")
        if len(parts) >= 3 and parts[1] == "en":
            # v1ex-en-question-0001 -> v1ex-de-question-0001 (lineage)
            return f"{parts[0]}-{lang}-{'-'.join(parts[2:])}"
        return f"{id_prefix}-{lang}-{rec['intent']}-{len(kept) + 1:04d}"

    blocked = blocked_texts()
    out_path = out_dir / f"{lang}.jsonl"
    kept: list[dict] = []
    seen: set[str] = set()
    drops = Counter()
    flags = Counter()

    batch = 16
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        texts = [r["text"] for r in chunk]
        t0 = time.time()
        results = generate_batch(tokenizer, model, spec, texts)
        for rec, translated in zip(chunk, results, strict=True):
            text = translated.strip()
            reason, flag = heuristic_qa(text, rec["text"])
            key = text.casefold() if reason is None else ""
            if reason is None and key in blocked:
                reason = "collision"
            if reason is None and key in seen:
                reason = "duplicate"
            if reason is not None:
                drops[reason] += 1
                continue
            if flag:
                flags[flag] += 1
            seen.add(key)
            rid = row_id(rec)
            kept.append(
                {
                    "id": rid,
                    "template_id": rid,
                    "language": lang,
                    "text": text,
                    "intent": rec["intent"],
                    "source_type": "nllb_translated",
                    "review_status": "machine_translated",
                    "split": "train",
                    "source_id": rec["id"],
                }
            )
        elapsed = time.time() - t0
        rate = len(chunk) / elapsed if elapsed else 0
        remaining = (len(rows) - start - len(chunk)) / rate if rate else 0
        print(
            f"[{lang}] {min(start + batch, len(rows))}/{len(rows)} "
            f"kept={len(kept)} drops={dict(drops)} flags={dict(flags)} "
            f"eta={remaining / 60:.1f}m",
            flush=True,
        )

    with out_path.open("w", encoding="utf-8") as handle:
        for rec in kept:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(
        f"[{lang}] wrote {out_path} ({len(kept)} rows, "
        f"drops={dict(drops)}, flags={dict(flags)})",
        flush=True,
    )


def main() -> int:
    out_dir = OUT_DIR
    input_path = EN_CORPUS
    id_prefix = "nllb"
    langs: list[str] = []
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--out-dir" and i + 1 < len(argv):
            out_dir = ROOT / argv[i + 1]
            i += 2
            continue
        if arg == "--input" and i + 1 < len(argv):
            input_path = ROOT / argv[i + 1]
            i += 2
            continue
        if arg == "--id-prefix" and i + 1 < len(argv):
            id_prefix = argv[i + 1]
            i += 2
            continue
        if arg in ENGINES:
            langs.append(arg)
        i += 1
    if not langs:
        print("no valid languages; expected subset of", list(ENGINES))
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_corpus(input_path)
    print(f"source: {input_path} ({len(rows)} rows)", flush=True)
    for lang in langs:
        translate_language(lang, rows, out_dir, id_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
