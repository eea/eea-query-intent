#!/bin/bash
# One acceptance-QA language: QA pass plus overlap repair.
# Idempotent: skips when the final file already exists, or when the raw
# file is not complete yet (825 rows) - the next phase cycle retries.
lang="$1"
raw="data/acceptance/v1/$lang.raw.jsonl"
final="data/acceptance/v1/$lang.jsonl"
if [ -f "$final" ]; then
  echo "$lang final already present, skipping" > "/tmp/acc_qa_$lang.log"
  exit 0
fi
if [ ! -f "$raw" ] || [ "$(wc -l < "$raw")" -lt 825 ]; then
  echo "$lang raw incomplete, skipping" > "/tmp/acc_qa_$lang.log"
  exit 0
fi
uv run python scripts/gpt_acceptance_qa.py "$lang" > "/tmp/acc_qa_$lang.log" 2>&1
if [ -f "$final" ]; then
  uv run python scripts/repair_acceptance_overlaps.py "$lang" >> "/tmp/acc_qa_$lang.log" 2>&1
fi
