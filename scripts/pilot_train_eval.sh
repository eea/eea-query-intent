#!/bin/bash
# NLLB pilot: train the pilot SetFit model on the translated mix, then
# run the frozen 28-language acceptance exam (raw + gated at 0.98),
# the threshold sweep, and the formal per-language report.
set -uo pipefail
cd /Users/razvan/Work/eea-query-intent
exec >> /tmp/pilot_train_eval.log 2>&1

echo "=== pilot train+eval started $(date) ==="
uv run python scripts/train_setfit.py \
  --train-file data/pilot/nllb/train.jsonl \
  --calibration-file data/pilot/nllb/calibration.jsonl \
  --model-dir models/setfit-pilot-nllb \
  --model-version setfit-v4-pilot-nllb \
  || { echo "TRAIN FAILED"; exit 1; }

uv run python scripts/predict_acceptance.py \
  --model models/setfit-pilot-nllb --threshold 0.0 --device mps \
  --out models/setfit-pilot-nllb/acceptance-predictions.jsonl \
  || { echo "PREDICT RAW FAILED"; exit 1; }

uv run python scripts/predict_acceptance.py \
  --model models/setfit-pilot-nllb --threshold 0.98 --device mps \
  --out models/setfit-pilot-nllb/acceptance-predictions-gated.jsonl \
  || { echo "PREDICT GATED FAILED"; exit 1; }

uv run python scripts/sweep_acceptance.py \
  --gold data/acceptance/v1/test.jsonl \
  --preds models/setfit-pilot-nllb/acceptance-predictions.jsonl \
  --thresholds 0.80,0.85,0.90,0.95,0.98,0.99 > reports/pilot_nllb_sweep.txt 2>&1

uv run eea-query-intent evaluate \
  --gold data/acceptance/v1/test.jsonl \
  --predictions models/setfit-pilot-nllb/acceptance-predictions-gated.jsonl \
  > reports/pilot_nllb_acceptance_report.json 2>&1 || true

uv run python scripts/pilot_compare.py \
  reports/final_acceptance_report.json \
  reports/pilot_nllb_acceptance_report.json > reports/pilot_nllb_comparison.txt 2>&1

echo "=== pilot train+eval DONE $(date) ==="
