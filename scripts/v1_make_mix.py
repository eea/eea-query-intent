"""Build the iteration-v1 training mix, calibration, and canonical exam.

Implements docs/experiments/2026-09-17-iteration-v1-preregistration.md:
strata assembly (A in-house native, B GPT-native raw, C MT sl/sv,
C2 legacy v3 rows with retention rules, E short bank), casefold dedup
with stratum priority, cross-label conflict resolution by surface form,
deterministic de-marking (~75% of question-marked question rows lose the
mark, every 4th keeps it), full lowercasing of the mix, calibration
assembly (old deduped + new short pool, lowercased), and the canonical
exam (frozen v1 rows untouched + new short pool, original case).

All input files are read-only; outputs go to data/pilot/v1/ and
data/acceptance/v2/. A manifest with input hashes, per-stratum counts,
conflict resolutions, and de-marking stats is written next to the mix.

Usage:
  uv run python scripts/v1_make_mix.py
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V1 = ROOT / "data" / "training" / "v1"
OUT = ROOT / "data" / "pilot" / "v1"
EXAM_OUT = ROOT / "data" / "acceptance" / "v2"
EXAM_V1 = ROOT / "data" / "acceptance" / "v1" / "test.jsonl"
BASE_MIX = ROOT / "data" / "expanded_v3" / "train.jsonl"
CALIB_OLD = ROOT / "data" / "expanded_v3" / "calibration.jsonl"
BANK = ROOT / "data" / "pilot" / "noq-short"
POOL = ROOT / "data" / "pilot" / "v1-pools"
HYGIENE = ROOT / "data" / "hygiene"
DROP_DECISIONS = HYGIENE / "2026-09-17" / "drop_decisions.jsonl"

# dataset.py REQUIRED_FIELDS contract: pool rows carry pipeline-local
# source_type values; map them into the package's allowed set. Translated
# and authored pool rows went through heuristic QA + Gemma blind judging +
# assistant adjudication, recorded as llm_reviewed per the project
# decision that LLM QA counts as review.
CONTRACT_SOURCE = {
    "nllb_translated": "synthetic_translated",
    "opus_translated": "synthetic_translated",
    "v1_exam_authored": "synthetic_generated",
    "v1_calib_authored": "synthetic_generated",
}

INHOUSE = [
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
]
GPT_NATIVE = ["cs", "el", "et", "hu", "lt", "lv"]
MT = ["sl", "sv"]
CLOSED = ["ga", "is", "mt"]
STARVED = GPT_NATIVE + MT
LEGACY_LANGS = STARVED + CLOSED  # stratum C2 (starved) + D (closed) legacy rows
ALL_LANGS = INHOUSE + GPT_NATIVE + MT + CLOSED

EXPECTED_SCRIPT = {"bg": "cyrillic", "el": "greek"}
LATIN = (
    frozenset(range(0x41, 0x7F))
    | frozenset(range(0xC0, 0x180))
    | frozenset(range(0x1E00, 0x1F00))
)
CYRILLIC = frozenset(range(0x400, 0x500))
GREEK = frozenset(range(0x370, 0x400)) | frozenset(range(0x1F00, 0x2000))
INVISIBLE_RE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]"
)
GARBLED = set("•\ufffd\u25a0\u25ab")

STRATUM_PRIORITY = {"A": 0, "B": 0, "C": 2, "C2": 3, "E": 4}


def load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def script_of(ch: int) -> str:
    if ch in CYRILLIC:
        return "cyrillic"
    if ch in GREEK:
        return "greek"
    if ch in LATIN:
        return "latin"
    return "other"


def legacy_ok(rec: dict) -> bool:
    """Retention check for a legacy v3 row (GPT review: same hygiene bar)."""
    text = (rec.get("text") or "").strip()
    lang = rec.get("language", "")
    if not text or INVISIBLE_RE.search(text):
        return False
    bad = sum(1 for ch in text if ch in GARBLED)
    if bad >= 3 or bad / max(len(text), 1) > 0.05:
        return False
    if len(text.split()) > 20 or len(text) > 500:
        return False
    if lang and rec.get("intent") != "unknown":
        expected = EXPECTED_SCRIPT.get(lang, "latin")
        letters = [ord(c) for c in text if c.isalpha()]
        if letters:
            hits = sum(1 for o in letters if script_of(o) == expected)
            if hits / len(letters) < 0.5:
                return False
    return True


def normalize(rec: dict, stratum: str) -> dict:
    return {
        "id": rec["id"],
        "intent": rec["intent"],
        "language": rec["language"],
        "text": rec["text"].strip(),
        "stratum": stratum,
        "source_id": rec.get("source_id") or rec.get("template_id"),
        "source_type": rec.get("source_type", "legacy"),
    }


def has_mark(text: str, lang: str) -> bool:
    t = text.rstrip()
    if not t:
        return False
    return t[-1] in ("?", ";") if lang == "el" else t.endswith("?")


def load_drop_decisions() -> tuple[set[str], set[tuple[str, str]], set[str]]:
    """Adjudicated rows to exclude (Gemma flags + assistant review).

    Returns (by_id, by_lang_text, by_source_id). source_id-level drops
    quarantine a concept across all its translations (the English source
    rows carry no source_id and are unaffected).
    """
    by_id: set[str] = set()
    by_text: set[tuple[str, str]] = set()
    by_source: set[str] = set()
    if DROP_DECISIONS.exists():
        for line in DROP_DECISIONS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("id"):
                by_id.add(rec["id"])
            if rec.get("source_id"):
                by_source.add(rec["source_id"])
            if rec.get("language") and rec.get("text"):
                by_text.add((rec["language"], rec["text"].casefold()))
    return by_id, by_text, by_source


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    EXAM_OUT.mkdir(parents=True, exist_ok=True)

    drop_ids, drop_texts, drop_sources = load_drop_decisions()
    manifest: dict = {"inputs": {}, "strata": {}, "resolutions": []}
    manifest["drop_decisions"] = {
        "by_id": len(drop_ids),
        "by_text": len(drop_texts),
        "by_source": len(drop_sources),
    }

    def is_dropped(rec: dict, lang: str, text: str) -> bool:
        return (
            rec.get("id") in drop_ids
            or rec.get("source_id") in drop_sources
            or (lang, text.casefold()) in drop_texts
        )

    def add(stratum: str, lang: str, path: Path, rows: list[dict], note: str = ""):
        manifest["inputs"][f"{stratum}/{lang}/{path.name}"] = sha256_16(path)
        manifest["strata"].setdefault(stratum, Counter())[lang] += len(rows)
        print(f"  {stratum:3} {lang:3} {path.name:28} {len(rows):6} {note}")

    # ------------------------------------------------------------ strata A: in-house
    print("stratum A: in-house native corpora")
    rows_a: list[dict] = []
    for lang in INHOUSE:
        p = V1 / f"{lang}.jsonl"
        rs = load(p)
        add("A", lang, p, rs)
        rows_a.extend(normalize(r, "A") for r in rs)

    # ------------------------------------------------ stratum B: GPT native raw
    print("stratum B: GPT-native raw corpora (hygiene-QA'd this iteration)")
    rows_b: list[dict] = []
    for lang in GPT_NATIVE:
        p = V1 / f"{lang}.raw.jsonl"
        rs = load(p)
        add("B", lang, p, rs)
        rows_b.extend(normalize(r, "B") for r in rs)

    # ------------------------------------------------------------ stratum C: MT sl/sv
    print("stratum C: NLLB machine translation")
    rows_c: list[dict] = []
    for lang in MT:
        p = V1 / f"{lang}.jsonl"
        if not p.exists():
            raise SystemExit(f"missing MT corpus: {p}")
        rs = load(p)
        add("C", lang, p, rs)
        rows_c.extend(normalize(r, "C") for r in rs)

    # ------------------------------------------------------------ stratum C2: legacy v3
    print("stratum C2: legacy v3 rows (retention-filtered)")
    legacy = [r for r in load(BASE_MIX) if r.get("language") in LEGACY_LANGS]
    base_path_ok = BASE_MIX.exists()
    if base_path_ok:
        manifest["inputs"]["C2/expanded_v3/train.jsonl"] = sha256_16(BASE_MIX)
    kept_c2: list[dict] = []
    dropped_c2 = Counter()
    for r in legacy:
        lang = r["language"]
        if not legacy_ok(r):
            dropped_c2[lang] += 1
            continue
        kept_c2.append(normalize(r, "C2"))
    manifest["strata"]["C2"] = Counter(r["language"] for r in kept_c2)
    manifest["c2_dropped"] = dict(dropped_c2)
    print(
        f"  C2  legacy rows: {len(legacy)} -> kept {len(kept_c2)} "
        f"(dropped {dict(dropped_c2)})"
    )
    rows_c2 = kept_c2

    # ------------------------------------------------------------ stratum E: short bank
    print("stratum E: short-question bank (170 rows/language)")
    rows_e: list[dict] = []
    for lang in ALL_LANGS:
        p = BANK / f"{lang}.jsonl"
        rs = load(p)
        add("E", lang, p, rs)
        rows_e.extend(normalize(r, "E") for r in rs)

    # ------------------------------------------------------------ dedup + conflicts
    def dropped(r: dict) -> bool:
        return is_dropped(r, r["language"], r["text"])

    all_strata = rows_a + rows_b + rows_c + rows_c2 + rows_e
    all_rows = [r for r in all_strata if not dropped(r)]
    manifest["drop_decisions_applied"] = len(all_strata) - len(all_rows)
    all_rows.sort(key=lambda r: STRATUM_PRIORITY[r["stratum"]])
    groups: dict[tuple[str, str], list[dict]] = {}
    order: list[tuple[str, str]] = []
    for r in all_rows:
        key = (r["language"], r["text"].casefold())
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)

    manifest["pre_dedup"] = len(all_rows)
    manifest["unique_texts"] = len(order)
    dedup_dropped = 0
    conflict_resolved = 0
    chosen: list[dict] = []
    for key in order:
        group = groups[key]
        if len(group) == 1:
            chosen.append(group[0])
            continue
        intents = {r["intent"] for r in group}
        if len(intents) == 1:
            chosen.append(group[0])
            dedup_dropped += len(group) - 1
            continue
        lang = key[0]
        text_cf = key[1]
        marked = text_cf.endswith("?") or (lang == "el" and text_cf.endswith(";"))
        if marked:
            wanted = "question"
        else:
            wanted = None
            for cand in ("retrieval", "exploratory", "claim"):
                if cand in intents:
                    wanted = cand
                    break
        same = [r for r in group if r["intent"] == wanted]
        kept = same[0] if same else group[0]
        conflict_resolved += 1
        dedup_dropped += len(group) - 1
        chosen.append(kept)
        manifest["resolutions"].append(
            {
                "language": lang,
                "text": text_cf[:120],
                "intents": sorted(intents),
                "kept": kept["intent"],
                "kept_id": kept["id"],
            }
        )
    manifest["dedup_dropped"] = dedup_dropped
    manifest["conflicts_resolved"] = conflict_resolved
    print(
        f"dedup+conflict: {len(all_rows)} -> {len(chosen)} "
        f"(dropped {dedup_dropped}, conflicts resolved {conflict_resolved})"
    )

    # ------------------------------------------------------------ de-marking
    marked_q: dict[str, list[dict]] = {}
    for r in chosen:
        if r["intent"] == "question" and has_mark(r["text"], r["language"]):
            marked_q.setdefault(r["language"], []).append(r)
    demarked = 0
    kept_marked = 0
    for _lang, rs in marked_q.items():
        for i, r in enumerate(rs):
            if i % 4 == 0:
                kept_marked += 1
            else:
                r["text"] = r["text"].rstrip()
                if r["text"].endswith(("?", ";")):
                    r["text"] = r["text"][:-1].rstrip()
                demarked += 1
    manifest["demarked"] = demarked
    manifest["kept_marked"] = kept_marked
    print(f"de-marking: {demarked} stripped, {kept_marked} kept (every 4th)")

    # ------------------------------------------------------------ lowercase + write mix
    for r in chosen:
        r["text"] = r["text"].casefold()
    mix_path = OUT / "train.jsonl"
    with mix_path.open("w", encoding="utf-8") as fh:
        for r in chosen:
            fh.write(
                json.dumps(
                    {
                        k: r[k]
                        for k in (
                            "id",
                            "intent",
                            "language",
                            "text",
                            "source_id",
                            "source_type",
                        )
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    manifest["mix_rows"] = len(chosen)
    manifest["mix_sha256_16"] = sha256_16(mix_path)
    per_lang = Counter(r["language"] for r in chosen)
    per_class = Counter(r["intent"] for r in chosen)
    manifest["mix_per_language"] = dict(sorted(per_lang.items()))
    manifest["mix_per_class"] = dict(sorted(per_class.items()))
    print(f"mix: {mix_path} ({len(chosen)} rows)")

    # ------------------------------------------------------------ calibration
    cal_old = load(CALIB_OLD)
    manifest["inputs"]["calibration-old"] = sha256_16(CALIB_OLD)
    seen_cal: set[str] = set()
    cal_rows: list[dict] = []
    cal_dup = 0
    for r in cal_old:
        key = (r["language"], r["text"].casefold())
        if key in seen_cal:
            cal_dup += 1
            continue
        seen_cal.add(key)
        cal_rows.append(
            {
                k: r[k]
                for k in (
                    "id",
                    "intent",
                    "language",
                    "text",
                    "template_id",
                    "source_type",
                    "review_status",
                    "split",
                )
            }
        )
    cal_pool_new = 0
    cal_drop_decisions = 0
    for lang in ALL_LANGS:
        p = POOL / "calib" / f"{lang}.jsonl"
        if not p.exists() and lang != "en":
            raise SystemExit(f"missing calib pool: {p}")
        src = p if p.exists() else POOL / "en_calib_short.jsonl"
        if lang == "en" and not p.exists():
            src = POOL / "en_calib_short.jsonl"
        for r in load(src):
            if r["language"] != lang:
                continue
            if is_dropped(r, lang, r["text"]):
                cal_drop_decisions += 1
                continue
            key = (lang, r["text"].casefold())
            if key in seen_cal:
                cal_dup += 1
                continue
            seen_cal.add(key)
            cal_rows.append(
                {
                    "id": r["id"],
                    "intent": r["intent"],
                    "language": lang,
                    "text": r["text"],
                    "template_id": r.get("template_id") or r["id"],
                    "source_type": CONTRACT_SOURCE.get(
                        r.get("source_type", ""), "synthetic_generated"
                    ),
                    "review_status": "llm_reviewed",
                    "split": "calibration",
                }
            )
            cal_pool_new += 1
    for r in cal_rows:
        r["text"] = r["text"].casefold()
        r["split"] = "calibration"
    cal_path = OUT / "calibration.jsonl"
    with cal_path.open("w", encoding="utf-8") as fh:
        for r in cal_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest["calibration_rows"] = len(cal_rows)
    manifest["calibration_new_pool_rows"] = cal_pool_new
    manifest["calibration_dedup_dropped"] = cal_dup
    manifest["calibration_drop_decisions"] = cal_drop_decisions
    manifest["calibration_sha256_16"] = sha256_16(cal_path)
    print(
        f"calibration: {cal_path} ({len(cal_rows)} rows, "
        f"new pool {cal_pool_new}, dedup dropped {cal_dup})"
    )

    # ------------------------------------------------------------ canonical exam
    exam_old = load(EXAM_V1)
    manifest["inputs"]["exam-v1"] = sha256_16(EXAM_V1)
    exam_rows = list(exam_old)
    seen_exam = {(r["language"], r["text"].casefold()) for r in exam_old}
    exam_new = 0
    exam_drop_decisions = 0
    for lang in ALL_LANGS:
        p = POOL / "exam" / f"{lang}.jsonl"
        if not p.exists() and lang != "en":
            raise SystemExit(f"missing exam pool: {p}")
        src = p if lang != "en" else POOL / "en_exam_short.jsonl"
        for r in load(src):
            if r["language"] != lang:
                continue
            if is_dropped(r, lang, r["text"]):
                exam_drop_decisions += 1
                continue
            key = (lang, r["text"].casefold())
            if key in seen_exam:
                continue
            seen_exam.add(key)
            out = {
                "id": r["id"],
                "intent": r["intent"],
                "language": lang,
                "text": r["text"],
                "template_id": r.get("template_id") or r["id"],
                "source_type": CONTRACT_SOURCE.get(
                    r.get("source_type", ""), "synthetic_generated"
                ),
                "review_status": "llm_reviewed",
                "split": "test",
            }
            exam_rows.append(out)
            exam_new += 1
    exam_path = EXAM_OUT / "test.jsonl"
    with exam_path.open("w", encoding="utf-8") as fh:
        for r in exam_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    exam_manifest = {
        "version": "v1-iteration-1",
        "base": "data/acceptance/v1/test.jsonl (21280 rows, untouched)",
        "base_sha256_16": sha256_16(EXAM_V1),
        "added_rows": exam_new,
        "total_rows": len(exam_rows),
        "sha256_16": sha256_16(exam_path),
        "note": (
            "new short rows (90 concepts/language, NLLB-translated, "
            "original case) are the diagnostic short-query slice; gold "
            "labels come from the English source concept class"
        ),
    }
    (EXAM_OUT / "manifest.json").write_text(
        json.dumps(exam_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest["exam_total"] = len(exam_rows)
    manifest["exam_new"] = exam_new
    manifest["exam_drop_decisions"] = exam_drop_decisions
    manifest["exam_sha256_16"] = sha256_16(exam_path)
    print(f"exam: {exam_path} ({len(exam_rows)} rows, {exam_new} new)")

    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"manifest: {manifest_path}")
    nfc = sum(1 for r in chosen if r["text"] != unicodedata.normalize("NFC", r["text"]))
    print(f"sanity: nfc-diff rows in mix = {nfc}")


if __name__ == "__main__":
    main()
