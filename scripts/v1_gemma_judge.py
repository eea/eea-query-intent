"""Gemma blind intent judge for the v1 data-hygiene pass (no GPT, no humans).

Pre-registration section 7.3: Gemma predicts each judged row's intent
BEFORE seeing the stored label. A disagreement (predicted != stored) is a
flag for assistant adjudication. Gemma is a flagging layer only — it never
decides keep/drop itself.

Judged set (deterministic, seed 17):
  - all severity-flagged rows from the hygiene scan except by-design ones
    (garbled, invisible, nfc, too_long, script_mismatch, dupe_xlang);
    qmark_nonquestion sampled at 100 rows
  - stratified random samples: 30/language from GPT-native raw (stratum B),
    30/language from MT sl/sv (stratum C), 15/language from in-house (A),
    10/language from the short bank (E), 10/language from each translated
    pool (exam + calib)

Batches of 20 rows per pi call (in-house Gemma, the free model), JSON
array replies, resumable via a progress file.

Usage: uv run python scripts/v1_gemma_judge.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from gpt_acceptance_gen import _pi_once  # noqa: E402

V1 = ROOT / "data" / "training" / "v1"
BANK = ROOT / "data" / "pilot" / "noq-short"
POOL = ROOT / "data" / "pilot" / "v1-pools"
HYGIENE = ROOT / "data" / "hygiene" / "2026-09-17"
OUT = HYGIENE / "gemma_judged.jsonl"
PROGRESS = ROOT / ".pipeline" / "v1_gemma_judge_progress"
GEMMA = "EEA/Inhouse-LLM/gemma-4-31B-it"
BATCH = 30
SEED = 17
INTENTS = ("question", "exploratory", "claim", "retrieval", "unknown")

SEVERITY_FLAGS = {
    "garbled",
    "invisible",
    "nfc",
    "too_long",
    "script_mismatch",
    "dupe_xlang",
}
QMARK_SAMPLE = 100


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def judge_item(corpus: str, rec: dict, why: str) -> dict:
    return {
        "corpus": corpus,
        "id": rec.get("id", "?"),
        "language": rec.get("language", ""),
        "stored": rec.get("intent", ""),
        "text": rec.get("text", ""),
        "why": why,
    }


def build_judged() -> list[dict]:
    rnd = random.Random(SEED)
    items: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    def add(corpus: str, rec: dict, why: str) -> None:
        key = (corpus, rec.get("id", "?"), rec.get("text", "").casefold())
        if key in seen or not rec.get("text", "").strip():
            return
        if rec.get("intent") not in INTENTS:
            return
        seen.add(key)
        items.append(judge_item(corpus, rec, why))

    # 1) severity flags from the scan
    flagged = load(HYGIENE / "flagged_rows.jsonl")
    qmark: list[dict] = []
    for rec in flagged:
        flags = set(rec.get("flags", []))
        if "label_conflict" in flags:
            continue  # resolved deterministically by the mix builder
        if "qmark_nonquestion" in flags and not (flags - {"qmark_nonquestion"}):
            qmark.append(rec)
            continue
        severity = flags & SEVERITY_FLAGS
        if severity:
            add(rec.get("corpus", "?"), rec, ",".join(sorted(severity)))
    sample_q = rnd.sample(qmark, min(QMARK_SAMPLE, len(qmark)))
    for rec in sample_q:
        add(rec.get("corpus", "?"), rec, "qmark_nonquestion(sample)")

    # 2) stratified samples
    for lang in ("cs", "el", "et", "hu", "lt", "lv"):
        rows = load(V1 / f"{lang}.raw.jsonl")
        for rec in rnd.sample(rows, min(30, len(rows))):
            add(f"B/{lang}", rec, "sample")
    for lang in ("sl", "sv"):
        rows = load(V1 / f"{lang}.jsonl")
        for rec in rnd.sample(rows, min(30, len(rows))):
            add(f"C/{lang}", rec, "sample")
    inhouse = [
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
    for lang in inhouse:
        rows = load(V1 / f"{lang}.jsonl")
        for rec in rnd.sample(rows, min(15, len(rows))):
            add(f"A/{lang}", rec, "sample")
    for p in sorted(BANK.glob("*.jsonl")):
        if p.name.startswith("counter-") or p.name in (
            "train.jsonl",
            "calibration.jsonl",
        ):
            continue
        rows = load(p)
        for rec in rnd.sample(rows, min(10, len(rows))):
            add(f"E/{p.stem}", rec, "sample")
    for pool in ("exam", "calib"):
        for p in sorted((POOL / pool).glob("*.jsonl")):
            rows = load(p)
            for rec in rnd.sample(rows, min(10, len(rows))):
                add(f"pool/{pool}/{p.stem}", rec, "sample")
    return items


PROMPT_HEAD = """You are classifying short search-box queries for the European
Environment Agency website. For each query, choose exactly one label:
- question: a direct question (usually ends with ?)
- exploratory: a request to learn about a topic (tell me about X, explain X)
- claim: a statement asserting something (X is bad, X should be done)
- retrieval: a keyword search or data lookup (names, stats, maps, years)
- unknown: off-topic for an EU environment site, or not a search query

Reply with ONLY a JSON array, one object per query, in the same order:
[{"i": 1, "intent": "retrieval"}, ...]

"""


def _parse_array(text: str) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text).strip()
    text = re.sub(r"\s*```$", "", text)
    a = text.find("[")
    b = text.rfind("]")
    if a == -1 or b == -1 or b < a:
        raise ValueError(f"no JSON array in model output: {text[:120]!r}")
    return json.loads(text[a : b + 1])


def judge_batch(items: list[dict], start: int) -> dict[int, str]:
    lines = []
    for j, it in enumerate(items[start : start + BATCH], 1):
        lines.append(f"{j}. [{it['language']}] {it['text']}")
    prompt = PROMPT_HEAD + "\n".join(lines)
    last_err: Exception | None = None
    for _attempt in range(3):
        text = _pi_once(prompt, GEMMA)
        try:
            data = _parse_array(text)
            preds: dict[int, str] = {}
            for row in data:
                i = int(row.get("i", row.get("id", 0)))
                intent = str(row.get("intent", "")).strip().lower()
                if 1 <= i <= len(items) - start and intent in INTENTS:
                    preds[i] = intent
            if preds:
                return preds
            last_err = ValueError(f"unusable reply: {text[:200]!r}")
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"batch at {start} failed: {last_err}")


def load_progress() -> dict[str, dict[int, str]]:
    done: dict[str, dict[int, str]] = {}
    if PROGRESS.exists():
        for line in PROGRESS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[str(rec["start"])] = {
                    int(k): v for k, v in rec["verdicts"].items()
                }
    return done


def main() -> int:
    items = build_judged()
    if not items:
        print("nothing to judge (pools missing?)")
        return 1
    by_lang = Counter(it["language"] for it in items)
    print(f"judged set: {len(items)} rows; per-language: {dict(by_lang)}")

    done = load_progress()
    out_lines: list[str] = []
    start = 0
    while start < len(items):
        key = str(start)
        batch = items[start : start + BATCH]
        if key in done:
            preds = done[key]
        else:
            preds = judge_batch(items, start)
            with PROGRESS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"start": start, "verdicts": preds}) + "\n")
            print(f"batch {start}-{min(start + BATCH, len(items))} done", flush=True)
        for j, it in enumerate(batch, 1):
            pred = preds.get(j, "")
            agree = pred == it["stored"] if pred else None
            out_lines.append(
                json.dumps(
                    {**it, "predicted": pred, "agree": agree},
                    ensure_ascii=False,
                )
            )
        start += BATCH

    OUT.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    parsed = [json.loads(line) for line in out_lines]
    agree = sum(1 for p in parsed if p["agree"] is True)
    disagree = [p for p in parsed if p["agree"] is False]
    no_pred = sum(1 for p in parsed if p["agree"] is None)
    print(
        f"\njudged: {len(out_lines)}  agree: {agree}  "
        f"disagree: {len(disagree)}  no-prediction: {no_pred}"
    )
    with (HYGIENE / "gemma_disagreements.jsonl").open("w", encoding="utf-8") as fh:
        for d in disagree:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"out: {OUT}\ndisagreements: {HYGIENE / 'gemma_disagreements.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
