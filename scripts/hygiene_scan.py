"""Deterministic data-hygiene scan for iteration v1 (no humans, no LLM).

Implements step 1 of docs/experiments/2026-09-17-iteration-v1-preregistration.md
section 7: objective, reproducible checks over every corpus that feeds the
mix, calibration, or exam. Flags are advisory — final keep/drop is decided
after the Gemma blind-judge pass and assistant review (sections 7.3-7.5).

Checks (per row):
  nfc                text is not NFC-normalized
  invisible          control / bidi / zero-width characters present
  garbled            bullet / replacement / block chars (NLLB failure mode)
  script_mismatch    <50% of letters in the language's expected script
                     (unknown-intent rows exempt: off-language junk is by design)
  too_long           >20 words or >500 chars (service guard bounds)
  empty              blank text
  dupe_intra         casefold exact duplicate within the corpus
  dupe_xlang         casefold exact match in another language's corpus (>3 words)
  label_conflict     same text (same language) carrying different intents
  qmark_nonquestion  retrieval/unknown/claim/exploratory ending in a question mark
  unmarked_question  question row without a trailing question mark
  exam_collision     text collides with the frozen acceptance exam
  mt_not_translated  MT row (has source_id) identical to its English source

Usage:
  uv run python scripts/hygiene_scan.py
  (corpus list below; outputs to data/hygiene/<date>/)
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "data" / "hygiene"
EXAM = ROOT / "data" / "acceptance" / "v2" / "test.jsonl"  # v2 is a superset of v1
EN_CORPUS = ROOT / "data" / "training" / "v1" / "en.jsonl"
V1 = ROOT / "data" / "training" / "v1"

INHOUSE = [
    "bg", "da", "de", "en", "es", "fi", "fr", "hr", "it", "nb", "nl", "nn",
    "pl", "pt", "ro", "sk", "tr",
]
GPT_RAW = ["cs", "el", "et", "hu", "lt", "lv"]

EXPECTED_SCRIPT = {"bg": "cyrillic", "el": "greek"}  # everything else: latin

INVISIBLE_RE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"
    "\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]"
)
GARBLED_CHARS = set("•\ufffd\u25a0\u25ab")
LATIN = frozenset(range(0x41, 0x7F)) | frozenset(range(0xC0, 0x180)) | frozenset(
    range(0x1E00, 0x1F00)
)
CYRILLIC = frozenset(range(0x400, 0x500))
GREEK = frozenset(range(0x370, 0x400)) | frozenset(range(0x1F00, 0x2000))


def script_of(ch: int) -> str:
    if ch in CYRILLIC:
        return "cyrillic"
    if ch in GREEK:
        return "greek"
    if ch in LATIN:
        return "latin"
    return "other"


def load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def corpus_list() -> list[tuple[str, Path]]:
    corpa: list[tuple[str, Path]] = []
    for lang in INHOUSE:
        p = V1 / f"{lang}.jsonl"
        if p.exists():
            corpa.append((f"v1/{lang}", p))
    for lang in GPT_RAW:
        p = V1 / f"{lang}.raw.jsonl"
        if p.exists():
            corpa.append((f"v1/{lang}-raw", p))
    for lang in ("sl", "sv"):
        p = V1 / f"{lang}.jsonl"
        if p.exists():
            corpa.append((f"v1/{lang}-mt", p))
    cal = ROOT / "data" / "pilot" / "v1" / "calibration.jsonl"
    if cal.exists():
        corpa.append(("calibration", cal))
    mix = ROOT / "data" / "pilot" / "v1" / "train.jsonl"
    if mix.exists():
        corpa.append(("v1-mix", mix))
    bank_dir = ROOT / "data" / "banks" / "v1-short"
    if bank_dir.exists():
        for p in sorted(bank_dir.glob("*.jsonl")):
            corpa.append((f"bank/{p.stem}", p))
    pools_dir = ROOT / "data" / "pilot" / "v1-pools"
    if pools_dir.exists():
        for p in sorted(pools_dir.glob("*/*.jsonl")):
            corpa.append((f"pool/{p.parent.name}/{p.stem}", p))
    return corpa


def check_row(
    rec: dict,
    lang: str | None,
    exam_blocked: set[str],
    en_by_id: dict[str, str],
    flags_out: list[str],
) -> None:
    text = rec.get("text", "")
    intent = rec.get("intent", "")

    if not text.strip():
        flags_out.append("empty")
        return
    if text != unicodedata.normalize("NFC", text):
        flags_out.append("nfc")
    if INVISIBLE_RE.search(text):
        flags_out.append("invisible")
    bad = sum(1 for ch in text if ch in GARBLED_CHARS)
    if bad >= 3 or bad / max(len(text), 1) > 0.05:
        flags_out.append("garbled")
    words = len(text.split())
    if words > 20 or len(text) > 500:
        flags_out.append("too_long")

    if lang and intent != "unknown":
        expected = EXPECTED_SCRIPT.get(lang, "latin")
        letters = [ord(c) for c in text if c.isalpha()]
        if letters:
            hits = sum(1 for o in letters if script_of(o) == expected)
            if hits / len(letters) < 0.5:
                flags_out.append("script_mismatch")

    last = text.rstrip()[-1:]
    qm = last in ("?", ";") if lang == "el" else text.rstrip().endswith("?")
    if qm and intent in ("retrieval", "unknown", "claim", "exploratory"):
        flags_out.append("qmark_nonquestion")
    if intent == "question" and not qm:
        flags_out.append("unmarked_question")

    folded = text.casefold()
    if folded in exam_blocked:
        flags_out.append("exam_collision")

    source_id = rec.get("source_id")
    if source_id and source_id in en_by_id:
        src = en_by_id[source_id]
        if folded == src.strip().casefold() and len(folded.split()) >= 4:
            flags_out.append("mt_not_translated")


def main() -> int:
    stamp = date.today().isoformat()
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    exam_blocked = {
        json.loads(line)["text"].casefold()
        for line in EXAM.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    en_by_id = {
        r["id"]: r["text"] for r in load_rows(EN_CORPUS)
    }

    corpa = corpus_list()
    report: dict[str, dict] = {}
    flagged_path = out_dir / "flagged_rows.jsonl"
    flagged = flagged_path.open("w", encoding="utf-8")

    # index for cross-corpus checks: folded text -> [(corpus, lang, intent, id)]
    index: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)

    for name, path in corpa:
        rows = load_rows(path)
        name_lang = name.split("/")[-1].split("-")[0]
        counts: Counter[str] = Counter()
        seen_local: dict[str, str] = {}
        # first pass: index + intra-corpus checks
        for rec in rows:
            text = rec.get("text", "").strip()
            folded = text.casefold()
            intent = rec.get("intent", "")
            rid = rec.get("id", "?")
            lang = rec.get("language") or name_lang
            if lang and folded:
                index[folded].append((name, lang, intent, rid))
            flags: list[str] = []
            check_row(rec, lang, exam_blocked, en_by_id, flags)
            if folded and lang:
                if folded in seen_local:
                    flags.append("dupe_intra")
                else:
                    seen_local[folded] = rid
            for f in flags:
                counts[f] += 1
            if flags:
                flagged.write(
                    json.dumps(
                        {
                            "corpus": name,
                            "id": rid,
                            "language": lang,
                            "intent": intent,
                            "flags": flags,
                            "text": text[:200],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        # second pass: cross-corpus duplicate + label conflict (needs full index)
        for rec in rows:
            text = rec.get("text", "").strip()
            folded = text.casefold()
            if not folded or len(folded.split()) <= 3:
                continue
            lang = rec.get("language") or name_lang
            if not lang:
                continue
            others = [e for e in index[folded] if e[0] != name]
            extra: list[str] = []
            if others:
                langs = {e[1] for e in others if e[1]}
                if langs - {lang}:
                    extra.append("dupe_xlang")
            intents = {e[2] for e in index[folded] if e[1] == lang and e[2]}
            own_intent = rec.get("intent", "")
            if own_intent and intents - {own_intent}:
                extra.append("label_conflict")
            if extra:
                for f in set(extra):
                    counts[f] += 1
                flagged.write(
                    json.dumps(
                        {
                            "corpus": name,
                            "id": rec.get("id", "?"),
                            "language": lang,
                            "intent": own_intent,
                            "flags": extra,
                            "text": text[:200],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        report[name] = {
            "rows": len(rows),
            "sha256_16": sha256_of(path),
            "flags": dict(counts),
        }
        print(f"{name:28} rows={len(rows):6} flags={dict(counts) or '-'}",
              flush=True)

    flagged.close()
    report_path = out_dir / "scan_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = flagged_path.read_text(encoding="utf-8").splitlines()
    total = sum(1 for line in lines if line.strip())
    print(f"\nreport:  {report_path}")
    print(f"flagged: {flagged_path} ({total} flagged-row entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
