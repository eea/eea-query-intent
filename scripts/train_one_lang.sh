#!/bin/bash
# Per-language 3000-row training corpus: generation, then QA, using the
# model routing decided by the confidence probe (data/train_routing.json /
# .pipeline/train_routing.env). GPT-routed languages (in-house model
# unconfident) run gen + QA on GPT Luna; the rest stay on the free
# in-house gateway.
#
# Idempotent: skips languages whose final file already has 3000 rows.
# Both steps are resumable (gen resumes per intent, QA checkpoints per
# batch), so a restart never loses completed work.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

LANG_="$1"
exec >> "/tmp/trn_1_${LANG_}.log" 2>&1

if [ -f "data/training/v1/${LANG_}.jsonl" ] && \
   [ "$(wc -l < "data/training/v1/${LANG_}.jsonl")" -ge 3000 ]; then
  echo "${LANG_}: already complete ($(wc -l < "data/training/v1/${LANG_}.jsonl") rows)"
  exit 0
fi

INHOUSE="EEA/Inhouse-LLM/gemma-4-31B-it"
GEN_M="$INHOUSE"
QA_M="$INHOUSE"
if [ -f .pipeline/train_routing.sh ]; then
  # shellcheck disable=SC1091
  source .pipeline/train_routing.sh
  # Indirect expansion: macOS /bin/bash is 3.2, which rejects nested
  # names like ${TRAIN_GEN_${LANG_}}. The indirection works there.
  GEN_V="TRAIN_GEN_${LANG_}"
  QA_V="TRAIN_QA_${LANG_}"
  GEN_M="${!GEN_V:-$INHOUSE}"
  QA_M="${!QA_V:-$INHOUSE}"
fi
export EEA_QI_GEN_MODEL="$GEN_M"
export EEA_QI_QA_MODEL="$QA_M"

echo "${LANG_} start $(date) gen_model=${GEN_M}"
uv run python scripts/gpt_train_gen.py "$LANG_" || { echo "${LANG_}: gen FAILED"; exit 1; }
echo "${LANG_} qa $(date) qa_model=${QA_M}"
uv run python scripts/gpt_train_qa.py "$LANG_" || { echo "${LANG_}: qa FAILED"; exit 1; }
echo "${LANG_} complete $(date) ($(wc -l < "data/training/v1/${LANG_}.jsonl") rows)"
