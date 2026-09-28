"""GPT-5.6-Sol QA pass over a language's raw TRAINING rows.

Mirror of the acceptance QA (scripts/gpt_acceptance_qa.py) for
data/training/v1: applies ok/fix/drop verdicts, tops up intents below
TRAIN_QUOTAS, prunes, and writes data/training/v1/<lang>.jsonl with
review_status llm_reviewed and split train. Final rows are re-checked
against every other dataset so a QA "fix" can never leak an acceptance
holdout text into training. The shared pipeline lives in
scripts/gpt_common.py.

Usage: uv run python scripts/gpt_train_qa.py <lang>
"""

import json
import sys
from pathlib import Path

from gpt_acceptance_qa import qa_batch
from gpt_common import (
    apply_verdicts,
    finalize_kept,
    load_progress,
    run_qa_batches,
    sanitize_kept,
    shortfall_splits,
    texts_from_dirs,
    top_up,
    write_report_log,
    write_shard,
)
from gpt_train_gen import DEDUP_DIRS, OUT_DIR, TRAIN_QUOTAS

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"

# Retrieval (1-6-word keyword phrases) has a lower diversity ceiling in
# the narrow EEA domain, so it gets a wider shortfall band than the
# other intents (gpt_common.shortfall_splits).
TOLERANCE = {"retrieval": 0.10}


def progress_path(lang: str) -> Path:
    # Deliberately NOT a .jsonl extension: corpus readers glob *.jsonl in
    # these data dirs and expect a "text" key on every row.
    return OUT_DIR / f"{lang}.qa_progress"


def final_avoid_set(exclude: frozenset[Path]) -> set[str]:
    dirs = [d for d in DEDUP_DIRS + [OUT_DIR] if d.exists()]
    return texts_from_dirs(dirs, exclude)


def main() -> None:
    lang = sys.argv[1]
    raw_path = OUT_DIR / f"{lang}.raw.jsonl"
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    progress = load_progress(progress_path(lang))
    verdicts: dict[int, dict] = {}
    for batch_verdicts in progress.values():
        verdicts.update(batch_verdicts)
    run_qa_batches(
        lang, rows, progress, verdicts, progress_path(lang), "train-QA", qa_batch
    )

    kept, stats = apply_verdicts(rows, verdicts, TRAIN_QUOTAS)

    # Compute the final safety pass's avoid set BEFORE the top-up so
    # top-up rows that collide with existing corpus texts are rejected at
    # generation time instead of being silently dropped at the end (that
    # caused below-target restart loops).
    avoid = final_avoid_set(
        frozenset({OUT_DIR / f"{lang}.jsonl", OUT_DIR / f"{lang}.raw.jsonl"})
    )
    sanitize_kept(kept, avoid)
    topups = top_up(lang, kept, avoid, 6, TRAIN_QUOTAS, qa_batch)

    final, leaked = finalize_kept(kept, TRAIN_QUOTAS, avoid)

    counts = {intent: len(texts) for intent, texts in final.items()}
    shortfall, tolerated, hard = shortfall_splits(counts, TRAIN_QUOTAS, TOLERANCE)
    log = {
        "language": lang,
        "raw_rows": len(rows),
        "verdicts": stats,
        "topups": topups,
        "leaked_dropped": leaked,
        "final_counts": counts,
        "shortfall": shortfall,
        "shortfall_tolerated": tolerated,
        "complete": not hard,
    }
    if tolerated:
        print(
            f"{lang}: WARNING shortfall below quota (tolerated): {tolerated}",
            flush=True,
        )
    if hard:
        # Do not write a truncated file: a partial output would look like a
        # finished dataset to the pipeline.
        write_report_log(REPORT_DIR, "train_qa", lang, log)
        print(f"{lang}: below target after QA: {counts}", flush=True)
        sys.exit(f"{lang}: below target after QA: {counts}")

    out_path = write_shard(OUT_DIR / f"{lang}.jsonl", lang, final, "trn", "train")
    write_report_log(REPORT_DIR, "train_qa", lang, log)
    print(f"{lang}: wrote {out_path} counts={counts}", flush=True)
    progress_path(lang).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
