#!/bin/bash
# Self-driving watcher: once the English training corpus is QA'd
# (data/training/v1/en.jsonl with all 3000 rows), run the full English
# validation experiment (retrain + acceptance sweep) and save the results.
# Retries up to 3 times on failure.
#
# Outputs:
#   reports/en_validation_<stamp>.log   full run log
#   reports/en_validation_sweep.txt     the threshold sweep table
#   .pipeline/en_validation_done.flag        completion marker (success only)
set -uo pipefail
cd "$(dirname "$0")/.."

while [ "$(wc -l < data/training/v1/en.jsonl 2>/dev/null || echo 0)" -lt 3000 ]; do sleep 120; done

for attempt in 1 2 3; do
  STAMP=$(date +%Y%m%d-%H%M)
  echo "en validation attempt $attempt $(date)"
  if bash scripts/run_en_validation.sh > "reports/en_validation_${STAMP}.log" 2>&1; then
    grep -B 1 -A 15 "worst FP" "reports/en_validation_${STAMP}.log" | tail -20 \
      > reports/en_validation_sweep.txt
    echo "DONE $(date)" >> reports/en_validation_sweep.txt
    touch .pipeline/en_validation_done.flag
    exit 0
  fi
  echo "attempt $attempt failed, retrying in 10 min" >> "reports/en_validation_${STAMP}.log"
  sleep 600
done
echo "FAILED after 3 attempts $(date)" > reports/en_validation_sweep.txt
exit 1
