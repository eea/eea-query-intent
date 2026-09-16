#!/bin/bash
# Iteration v1: train 3 fixed seeds of a candidate FAMILY, derive each
# seed's deployment threshold from the CALIBRATION set only (never the
# exam), and select the family's best seed on calibration.
#
# Families:
#   d1       five-class SetFit on MiniLM-L12 (the D-1-class baseline)
#   binary   objective-aligned binary head on MiniLM-L12
#   backbone five-class SetFit on intfloat/multilingual-e5-small
#
# The canonical exam is run exactly once, later, on the single locked
# finalist — see the pre-registration.
#
# Usage: bash scripts/v1_train_eval.sh [d1|binary|backbone]
set -uo pipefail
cd /Users/razvan/Work/eea-query-intent
FAMILY="${1:-d1}"
exec >> "/tmp/v1_train_eval_${FAMILY}.log" 2>&1

case "${FAMILY}" in
  d1)
    PREFIX="setfit-v1"
    EXTRA=()
    ;;
  binary)
    PREFIX="setfit-v1b"
    EXTRA=(--binary)
    ;;
  backbone)
    PREFIX="setfit-v1e5"
    EXTRA=(--backbone intfloat/multilingual-e5-small)
    ;;
  *)
    echo "unknown family: ${FAMILY}"
    exit 2
    ;;
esac

echo "=== v1 ${FAMILY} train+eval started $(date) ==="
rm -f ".pipeline/v1_${FAMILY}_train_done.flag"

for SEED in 1 2 3; do
  DIR="models/${PREFIX}-s${SEED}"
  if [ -f "${DIR}/manifest.json" ] && [ -f "${DIR}/calibration-predictions.jsonl" ]; then
    echo "=== ${FAMILY} seed ${SEED}: model exists, skipping train ($(date)) ==="
  else
  echo "=== ${FAMILY} seed ${SEED}: train start $(date) ==="
  uv run python scripts/train_setfit.py \
    --train-file data/pilot/v1/train.jsonl \
    --calibration-file data/pilot/v1/calibration.jsonl \
    --model-dir "${DIR}" \
    --model-version "setfit-v1-${FAMILY}" \
    --seed "${SEED}" \
        ${EXTRA[@]+"${EXTRA[@]}"} \
    || { echo "TRAIN FAILED ${FAMILY} seed ${SEED}"; exit 1; }
  fi

  # train_setfit.py already emitted <dir>/calibration-predictions.jsonl
  # (raw probabilities; input lowercased because the v1 calibration file
  # is lowercased). Derive the threshold from calibration only.
  uv run python scripts/choose_final_threshold.py \
    --gold data/pilot/v1/calibration.jsonl \
    --preds "${DIR}/calibration-predictions.jsonl" \
    --out "${DIR}/threshold.json" \
    || { echo "THRESHOLD FAILED ${FAMILY} seed ${SEED}"; exit 1; }
  echo "=== ${FAMILY} seed ${SEED}: done $(date) ==="
done

uv run python scripts/v1_select_seed.py "${FAMILY}" \
  || { echo "SELECTION FAILED ${FAMILY}"; exit 1; }

touch ".pipeline/v1_${FAMILY}_train_done.flag"
echo "=== v1 ${FAMILY} train+eval DONE $(date) ==="
