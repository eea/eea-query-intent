#!/bin/bash
# noq-short candidate: train setfit-v4-noq-short on the question-mark
# stripped + short-bank mix, then run the frozen 28-language acceptance
# exam (raw + gated at 0.98), the threshold sweep, and the formal
# per-language report + comparison vs the current setfit-v4 report.
set -uo pipefail
cd /Users/razvan/Work/eea-query-intent
exec >> /tmp/noq_train_eval.log 2>&1

echo "=== noq train+eval started $(date) ==="
uv run python scripts/train_setfit.py \
  --train-file data/pilot/noq-short/train.jsonl \
  --calibration-file data/pilot/noq-short/calibration.jsonl \
  --model-dir models/setfit-noq-short \
  --model-version setfit-v4-noq-short \
  || { echo "TRAIN FAILED"; exit 1; }

uv run python scripts/predict_acceptance.py \
  --model models/setfit-noq-short --threshold 0.0 --device mps --lowercase-input \
  --out models/setfit-noq-short/acceptance-predictions.jsonl \
  || { echo "PREDICT RAW FAILED"; exit 1; }

uv run python scripts/predict_acceptance.py \
  --model models/setfit-noq-short --threshold 0.98 --device mps --lowercase-input \
  --out models/setfit-noq-short/acceptance-predictions-gated.jsonl \
  || { echo "PREDICT GATED FAILED"; exit 1; }

uv run python scripts/sweep_acceptance.py \
  --gold data/acceptance/v1/test.jsonl \
  --preds models/setfit-noq-short/acceptance-predictions.jsonl \
  --thresholds 0.80,0.85,0.90,0.95,0.98,0.99 > reports/noq_short_sweep.txt 2>&1

uv run eea-query-intent evaluate \
  --gold data/acceptance/v1/test.jsonl \
  --predictions models/setfit-noq-short/acceptance-predictions-gated.jsonl \
  > reports/noq_short_acceptance_report.json 2>&1 || true

uv run python scripts/pilot_compare.py \
  reports/final_acceptance_report.json \
  reports/noq_short_acceptance_report.json > reports/noq_short_comparison.txt 2>&1

echo "=== noq train+eval DONE $(date) ==="
