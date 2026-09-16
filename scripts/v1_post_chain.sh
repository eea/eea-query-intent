#!/bin/bash
# Overnight post-chain: wait for the family chain, lock the finalist
# (calibration only), run the canonical v2 exam exactly once, and
# summarize. Stops here — promotion is a judgment step for the morning.
# Deterministic only; reads no user input.
set -uo pipefail
cd /Users/razvan/Work/eea-query-intent
exec >> /tmp/v1_post_chain.log 2>&1

echo "=== post-chain waiter started $(date) ==="
for i in $(seq 1 160); do  # ~13 h at 300 s
  [ -f .pipeline/v1_all_families_done.flag ] && break
  sleep 300
done
if [ ! -f .pipeline/v1_all_families_done.flag ]; then
  echo "family chain never finished within window"
  exit 1
fi

LOCK=$(uv run python scripts/v1_lock_finalist.py)
if [ $? -ne 0 ]; then
  echo "finalist lock FAILED"
  exit 1
fi
echo "$LOCK"
DIR=$(echo "$LOCK" | sed -n 's/^LOCKED=//p')
THR=$(echo "$LOCK" | sed -n 's/^THRESHOLD=//p')

mkdir -p reports/v1_overnight
cp models/finalist/lock.json reports/v1_overnight/ 2>/dev/null || true
cp models/setfit-v1*/selection.json reports/v1_overnight/ 2>/dev/null || true

if uv run python scripts/v1_canonical_exam.py "$DIR" "$THR"; then
  cp "reports/v1_canonical_exam_$(basename "$DIR").json" reports/v1_overnight/ 2>/dev/null || true
else
  echo "canonical exam FAILED - morning must investigate before promotion"
  touch .pipeline/v1_overnight_failed.flag
  exit 1
fi

echo "=== post-chain DONE $(date) ==="
touch .pipeline/v1_overnight_done.flag
