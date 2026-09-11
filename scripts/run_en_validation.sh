#!/usr/bin/env bash
# English validation stage for the diverse-data retrain.
#
# Requires data/training/v1/en.jsonl (from gpt_train_gen.py + gpt_train_qa.py).
# Trains an English-only SetFit head on the frozen English base + the new
# 3,000-row diverse corpus, predicts the English acceptance shard with NO
# abstention gating (threshold 0.0) so raw probabilities are kept, and
# sweeps the threshold table for comparison against setfit-v3.
#
# Usage: bash scripts/run_en_validation.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f data/training/v1/en.jsonl ]; then
  echo "missing data/training/v1/en.jsonl - run gpt_train_gen.py en + gpt_train_qa.py en first" >&2
  exit 1
fi

uv run python scripts/make_en_validation.py
uv run python scripts/train_setfit.py \
  --train-file data/en_validation/train.jsonl \
  --calibration-file data/en_validation/calibration.jsonl \
  --test-file data/acceptance/v1/en.jsonl \
  --model-dir models/setfit-en2 \
  --model-version setfit-en2
uv run python scripts/predict_acceptance.py \
  --input data/acceptance/v1/en.jsonl \
  --model models/setfit-en2 \
  --out models/setfit-en2/acceptance-predictions-en.jsonl \
  --threshold 0.0 \
  --device mps
uv run python scripts/sweep_acceptance.py \
  --gold data/acceptance/v1/en.jsonl \
  --preds models/setfit-en2/acceptance-predictions-en.jsonl \
  --thresholds 0.85,0.90,0.92,0.94,0.95,0.96,0.97,0.98,0.99,0.995,0.999
