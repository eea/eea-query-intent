#!/usr/bin/env bash
# Stage 3: generate the 27-language diverse training corpus on the free
# in-house gateway (Gemma 4 31B IT), then QA it on the same gateway.
# Launch AFTER the English validation (run_en_validation.sh)
# confirms the recipe closes the boundary gap.
#
# Generation runs at P4 (gateway parallelism verified at 4); the QA phase
# starts only after every generation finishes.
#
# Usage: bash scripts/run_training_batches.sh
set -uo pipefail
cd "$(dirname "$0")/.."

# Training data is study material: both generation and review run on the
# free in-house gateway (top-up regeneration included), not Codex quota.
export EEA_QI_GEN_MODEL="EEA/Inhouse-LLM/gemma-4-31B-it"
export EEA_QI_QA_MODEL="EEA/Inhouse-LLM/gemma-4-31B-it"

rm -f /tmp/trn_gen_done.flag /tmp/trn_qa_done.flag

# 27 non-English languages
uv run python -c "
from eea_query_intent.languages import SUPPORTED_LANGUAGE_CODES
print(' '.join(sorted(SUPPORTED_LANGUAGE_CODES - {'en'})))
" > /tmp/trn_langs.txt

nohup bash -c 'cat /tmp/trn_langs.txt | xargs -P 4 -I {} bash -c "uv run python scripts/gpt_train_gen.py {} > /tmp/trn_gen_{}.log 2>&1"; touch /tmp/trn_gen_done.flag' \
  > /tmp/trn_orchestrator.log 2>&1 &
echo "training generation launched (P4), pid $!"

# QA phase: waits for generation, then QAs each language
nohup bash -c '
while [ ! -f /tmp/trn_gen_done.flag ]; do sleep 60; done
for l in $(cat /tmp/trn_langs.txt); do
  [ -f data/training/v1/$l.raw.jsonl ] || { echo "skip $l (no raw file)" >> /tmp/trn_qa.log; continue; }
  uv run python scripts/gpt_train_qa.py "$l" > /tmp/trn_qa_$l.log 2>&1 || echo "QA FAILED: $l" >> /tmp/trn_qa.log
done
touch /tmp/trn_qa_done.flag
' > /tmp/trn_qa_phase.log 2>&1 &
echo "training QA phase armed, pid $!"
