#!/bin/bash
# Iteration v1: run all three candidate families sequentially.
# (MPS cannot safely parallelize model training on one GPU.)
# Per-family logs: /tmp/v1_train_eval_<family>.log
# Per-family flags: .pipeline/v1_<family>_train_done.flag
set -uo pipefail
cd /Users/razvan/Work/eea-query-intent
exec >> /tmp/v1_family_chain.log 2>&1

echo "=== family chain started $(date) ==="
rm -f .pipeline/v1_all_families_done.flag

for F in d1 binary backbone; do
  echo "--- launching family ${F} $(date) ---"
  bash scripts/v1_train_eval.sh "${F}"
  if [ -f ".pipeline/v1_${F}_train_done.flag" ]; then
    echo "--- family ${F} OK $(date) ---"
  else
    echo "--- family ${F} FAILED (continuing with next) $(date) ---"
  fi
done

touch .pipeline/v1_all_families_done.flag
echo "=== family chain DONE $(date) ==="
