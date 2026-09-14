#!/bin/bash
# Final build chain (relaunch_all.sh stage 7). Fires once all 28 training
# finals AND all 28 acceptance shards exist:
#   1. merge the 28 acceptance shards into data/acceptance/v1/test.jsonl
#   2. fold the 28x3000 training corpus into the expanded_v2 base
#   3. train the final SetFit head -> models/setfit (setfit-v4)
#   4. raw predictions on the 28-language acceptance set
#   5. threshold selection (smallest threshold with worst-language
#      no-AI false-positive rate <= 1%)
#   6. apply the threshold to the model manifest + formal evaluation
#   7. restart the local service so it serves the new model
#   8. summary -> reports/final_build_summary.txt + done flag
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"
exec >> /tmp/final_build.log 2>&1

echo "=== final build started $(date) ==="

uv run python scripts/validate_acceptance.py --merge \
  || { echo "acceptance merge FAILED"; exit 1; }
uv run python scripts/make_final_v3.py \
  || { echo "make_final_v3 FAILED"; exit 1; }
uv run python scripts/train_setfit.py \
  --train-file data/expanded_v3/train.jsonl \
  --calibration-file data/expanded_v3/calibration.jsonl \
  --model-dir models/setfit \
  --model-version setfit-v4 \
  || { echo "train FAILED"; exit 1; }
uv run python scripts/predict_acceptance.py \
  --model models/setfit --threshold 0.0 --device mps \
  --out models/setfit/acceptance-predictions.jsonl \
  || { echo "predict (raw) FAILED"; exit 1; }
uv run python scripts/choose_final_threshold.py \
  --gold data/acceptance/v1/test.jsonl \
  --preds models/setfit/acceptance-predictions.jsonl \
  || { echo "threshold selection FAILED"; exit 1; }

THR=$(uv run python -c "import json; print(json.load(open('reports/final_threshold.json'))['threshold'])")
uv run python scripts/apply_threshold.py --threshold "$THR" \
  || { echo "apply_threshold FAILED"; exit 1; }

uv run python scripts/sweep_acceptance.py \
  --gold data/acceptance/v1/test.jsonl \
  --preds models/setfit/acceptance-predictions.jsonl \
  --thresholds 0.80,0.85,0.90,0.95,0.98,0.99 > reports/final_sweep.txt 2>&1

# gated predictions at the chosen threshold (manifest already updated)
uv run python scripts/predict_acceptance.py \
  --model models/setfit \
  --out models/setfit/acceptance-predictions-gated.jsonl \
  || { echo "predict (gated) FAILED"; exit 1; }
uv run eea-query-intent evaluate \
  --gold data/acceptance/v1/test.jsonl \
  --predictions models/setfit/acceptance-predictions-gated.jsonl \
  > reports/final_acceptance_report.json 2>&1 || true

# restart the local service via its own launchd job (a nohup child of
# this build job would be torn down with the job)
echo "restarting local service for the new model (launchd job)"
launchctl remove com.razvan.eeaki-service 2>/dev/null
pkill -f "eea_query_intent.service" 2>/dev/null
sleep 3
launchctl submit -l com.razvan.eeaki-service \
  -o /tmp/qi-service.log -e /tmp/qi-service_err.log \
  -- /bin/bash "$(pwd)/scripts/run_service.sh"
sleep 30
curl -s --max-time 10 http://127.0.0.1:8100/health || echo "service health check failed"

{
  echo "final build $(date)"
  echo "chosen threshold: $THR (see reports/final_threshold.json)"
  echo ""
  echo "sweep:"
  cat reports/final_sweep.txt
  echo ""
  echo "formal report (chosen threshold):"
  head -60 reports/final_acceptance_report.json
} > reports/final_build_summary.txt
touch .pipeline/final_build_done.flag
echo "=== final build DONE $(date) ==="
