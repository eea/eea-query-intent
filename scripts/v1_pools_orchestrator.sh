#!/bin/bash
# v1 exam/calibration pool translation orchestrator.
#
# Waits for the sl/sv corpus translation (MPS) to finish, then translates
# the two English concept pools (data/pilot/v1-pools/) to all 27 non-English
# targets, then re-runs the deterministic hygiene scan over everything.
#
# Launch: nohup caffeinate -dis bash scripts/v1_pools_orchestrator.sh \
#   <sl/sv pid> > /tmp/v1_pools_orchestrator.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

SL_SV_PID="${1:?usage: v1_pools_orchestrator.sh <sl-sv-pid>}"

echo "waiting for sl/sv translation pid ${SL_SV_PID} ..."
while kill -0 "${SL_SV_PID}" 2>/dev/null; do
  sleep 20
done
echo "sl/sv translation done"

LANGS="bg cs da de el et fi fr ga hr hu is it lt lv mt nb nl nn pl pt ro sk sl sv tr"

echo "=== exam pool: 27 targets x 90 concepts ==="
uv run --with sentencepiece python scripts/pilot_translate_nllb.py ${LANGS} \
  --input data/pilot/v1-pools/en_exam_short.jsonl \
  --out-dir data/pilot/v1-pools/exam

echo "=== calib pool: 27 targets x 100 concepts ==="
uv run --with sentencepiece python scripts/pilot_translate_nllb.py ${LANGS} \
  --input data/pilot/v1-pools/en_calib_short.jsonl \
  --out-dir data/pilot/v1-pools/calib

echo "=== hygiene scan ==="
uv run python scripts/hygiene_scan.py

touch .pipeline/v1_pools_done.flag
echo "v1 pools translation DONE"
