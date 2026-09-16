#!/bin/bash
# noq-short pipeline orchestrator: waits for the background translation
# (pid passed as $1), then builds the candidate mix, trains the candidate
# model, and runs the frozen exam + sweep + comparison.
set -uo pipefail
cd /Users/razvan/Work/eea-query-intent
exec >> /tmp/noq_pipeline.log 2>&1

echo "=== noq pipeline started $(date) ==="

# 1. wait for BOTH translation passes (main + extra) via the sentinel flag
while [ ! -f .pipeline/noq_translate_done.flag ]; do
  sleep 30
done

# 2. all 28 bank files must exist (en authored + 27 translated)
for code in en bg da de es fi fr hr it nb nl nn pl pt ro sk tr \
            cs el et hu lt lv sl sv ga is mt; do
  if [ ! -f "data/pilot/noq-short/${code}.jsonl" ]; then
    echo "MISSING bank file for ${code} - aborting pipeline"
    exit 1
  fi
done

# 3. build the candidate mix (Fix A strip + Fix B bank append)
uv run python scripts/noq_make_mix.py || { echo "MIX FAILED"; exit 1; }

# 4. train + frozen exam + sweep + comparison
bash scripts/noq_train_eval.sh || { echo "TRAIN/EVAL FAILED"; exit 1; }

touch .pipeline/noq_pipeline_done.flag
echo "=== noq pipeline DONE $(date) ==="
