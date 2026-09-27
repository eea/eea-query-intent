"""GPT-5.6-Sol adversarial QA pass over a language's raw acceptance rows.

Reads ``data/acceptance/v1/<lang>.raw.jsonl`` (unreviewed, buffered
counts), asks GPT-5.6 Sol to review every row (label correctness, native
grammar, search-box register, duplicates), applies ok/fix/drop verdicts,
tops up any intent that falls below its target, prunes to the target
counts, renumbers ids, and writes the final
``data/acceptance/v1/<lang>.jsonl`` with ``review_status:
llm_reviewed``. The full verdict log goes to
``reports/acceptance_qa_<lang>.json``. The shared pipeline lives in
scripts/gpt_common.py.

Usage: uv run python scripts/gpt_acceptance_qa.py <lang>
"""

import json
import sys
from pathlib import Path

from gpt_acceptance_gen import NAMES, QA_MODEL, call_gpt_raw
from gpt_common import (
    apply_verdicts,
    finalize_kept,
    load_progress,
    run_qa_batches,
    sanitize_kept,
    shortfall_splits,
    texts_from_dir,
    top_up,
    write_report_log,
    write_shard,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "acceptance" / "v1"
REPORT_DIR = ROOT / "reports"

# A shortfall of up to 1% per intent is accepted with a loud warning;
# the evaluate CLI's count gates (300+ per side) are far below it, so
# the exam's statistics are unaffected.
DEFAULT_TOLERANCE = 0.01


def progress_path(lang: str) -> Path:
    # Deliberately NOT a .jsonl extension: corpus readers glob *.jsonl in
    # these data dirs and expect a "text" key on every row.
    return DATA_DIR / f"{lang}.qa_progress"


TARGETS = {
    "question": 150,
    "exploratory": 120,
    "claim": 90,
    "retrieval": 350,
    "unknown": 50,
}

QA_TMPL = """You are an adversarial native @NAME@ reviewer and a data-quality auditor. Below are numbered search-box queries for the European Environment Agency website, each with an intent label. For every row return exactly one verdict:
- "ok": the text is grammatical and natural for a native speaker, reads like a real search-box entry, and matches its label's definition.
- "fix": the intent is right but the text is ungrammatical, unnatural, not in native @NAME@, or borderline violates its definition in a fixable way. Provide the corrected text in "t".
- "drop": unusable (intent wrong beyond repair, broken or empty text, or a duplicate of another row in this batch).

Label definitions:
- question: @DEF_QUESTION@
- exploratory: @DEF_EXPLORATORY@
- claim: @DEF_CLAIM@
- retrieval: @DEF_RETRIEVAL@
- unknown: @DEF_UNKNOWN@

Extra checks: text must be in @NAME@ (Latin acronyms, proper nouns, digits are fine), at most 20 words, plausible as something a real visitor would type.

Output STRICT JSON only, no prose, no markdown: {"rows":[{"i":1,"v":"ok","t":"","r":"why"}]} - one entry per numbered row, in order. "t" is filled only for "fix"; "r" is at most 8 words.

Rows:
"""


def qa_batch(rows: list[tuple[int, str, str]], lang: str) -> dict:
    from gpt_acceptance_gen import DEFINITIONS

    prompt = (
        QA_TMPL.replace("@NAME@", NAMES[lang])
        .replace("@DEF_QUESTION@", DEFINITIONS["question"])
        .replace("@DEF_EXPLORATORY@", DEFINITIONS["exploratory"])
        .replace("@DEF_CLAIM@", DEFINITIONS["claim"])
        .replace("@DEF_RETRIEVAL@", DEFINITIONS["retrieval"])
        .replace("@DEF_UNKNOWN@", DEFINITIONS["unknown"])
    )
    for i, _intent, text in rows:
        prompt += f"{i} [{_intent}] {text}\n"
    return call_gpt_raw(prompt, model=QA_MODEL)


def corpus_avoid_set() -> set[str]:
    """Texts a new acceptance row must stay disjoint from (training data).

    Used to filter top-up rows at generation time: a top-up row that
    duplicates training text would be rejected at merge time, wasting
    the top-up slot.
    """
    seen: set[str] = set()
    for d in (
        DATA_DIR,
        ROOT / "data" / "acceptance" / "v2",
        ROOT / "data" / "pilot" / "v1",
        ROOT / "data" / "banks" / "v1-short",
    ):
        if not d.exists():
            continue
        seen.update(texts_from_dir(d))
    return seen


def main() -> None:
    lang = sys.argv[1]
    raw_path = DATA_DIR / f"{lang}.raw.jsonl"
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    progress = load_progress(progress_path(lang))
    verdicts: dict[int, dict] = {}
    for batch_verdicts in progress.values():
        verdicts.update(batch_verdicts)
    run_qa_batches(lang, rows, progress, verdicts, progress_path(lang), "QA", qa_batch)

    kept, stats = apply_verdicts(rows, verdicts, TARGETS)

    # top-up any intent below target (max 4 cycles); reject top-up rows
    # that collide with the training corpus so every slot gets a fresh row
    avoid = corpus_avoid_set()
    sanitize_kept(kept, avoid)
    topups = top_up(lang, kept, avoid, 4, TARGETS, qa_batch)

    final, _leaked = finalize_kept(kept, TARGETS)

    counts = {intent: len(texts) for intent, texts in final.items()}
    _shortfall, tolerated, hard = shortfall_splits(
        counts, TARGETS, None, DEFAULT_TOLERANCE
    )
    log = {
        "language": lang,
        "raw_rows": len(rows),
        "verdicts": stats,
        "topups": topups,
        "final_counts": counts,
        "shortfall_tolerated": tolerated,
        "complete": not hard,
    }
    if tolerated:
        print(
            f"{lang}: WARNING shortfall below quota (tolerated): {tolerated}",
            flush=True,
        )
    if hard:
        # Do not write a truncated shard: a partial output would look like a
        # finished acceptance set to the pipeline.
        write_report_log(REPORT_DIR, "acceptance_qa", lang, log)
        print(f"{lang}: below target after QA: {counts}", flush=True)
        sys.exit(f"{lang}: below target after QA: {counts}")

    out_path = write_shard(DATA_DIR / f"{lang}.jsonl", lang, final, "acc", "test")
    write_report_log(REPORT_DIR, "acceptance_qa", lang, log)
    print(f"{lang}: wrote {out_path} counts={counts}", flush=True)
    progress_path(lang).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
