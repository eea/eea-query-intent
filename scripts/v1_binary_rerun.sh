#!/bin/bash
# Second wave: once the first chain (d1/backbone) is done, complete the
# binary family (s1 model reused; s2+s3 trained with the fixed binary
# prediction writer). Deterministic; no user input.
set -uo pipefail
cd /Users/razvan/Work/eea-query-intent
exec >> /tmp/v1_binary_rerun.log 2>&1

echo "=== binary rerun waiter started $(date) ==="
for i in $(seq 1 160); do  # ~13 h at 300 s
  [ -f .pipeline/v1_all_families_done.flag ] && break
  sleep 300
done
if [ ! -f .pipeline/v1_all_families_done.flag ]; then
  echo "family chain never finished within window"
  exit 1
fi

echo "--- first chain done; running binary family $(date) ---"
rm -f .pipeline/v1_binary_rerun_done.flag
bash scripts/v1_train_eval.sh binary
if [ -f "models/setfit-v1b/selection.json" ]; then
  echo "--- binary rerun OK $(date) ---"
  touch .pipeline/v1_binary_rerun_done.flag
else
  echo "--- binary rerun FAILED $(date) ---"
  exit 1
fi
