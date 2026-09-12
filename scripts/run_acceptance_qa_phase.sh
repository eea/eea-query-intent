#!/bin/bash
# Acceptance QA phase: waits for the generation flag, then QAs each language
# (idempotently - languages that already have a final <lang>.jsonl are
# skipped). Runs at P4. Safe to launch repeatedly: a finished phase exits
# and a new one will find everything already done.
cd "$(dirname "$0")/.."

if [ ! -f .pipeline/acc_gen_done.flag ]; then
  echo "waiting for generation to finish..."
  while [ ! -f .pipeline/acc_gen_done.flag ]; do sleep 60; done
fi
echo "generation done, starting QA phase $(date)"
# guard: refuse to run with a broken language list (would touch the flag vacuously)
if [ ! -s .pipeline/acc_langs.txt ] || [ "$(tr -d ' ' < .pipeline/acc_langs.txt | wc -c)" -lt 54 ]; then
  echo "ERROR: language list missing or broken - exiting for retry"
  exit 1
fi
cat .pipeline/acc_langs.txt | xargs -P 4 -I {} bash -c '
  if [ -f data/acceptance/v1/{}.jsonl ]; then
    echo "{} final already present, skipping" > /tmp/acc_qa_{}.log
  else
    uv run python scripts/gpt_acceptance_qa.py {} > /tmp/acc_qa_{}.log 2>&1
    uv run python scripts/repair_acceptance_overlaps.py {} >> /tmp/acc_qa_{}.log 2>&1
  fi
'
touch .pipeline/acc_qa_done.flag
echo "QA phase done $(date)"
