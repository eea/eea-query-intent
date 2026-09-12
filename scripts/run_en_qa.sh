#!/bin/bash
# English training QA worker. Runs as a launchd job (see relaunch_all.sh
# stage 4). The QA pass itself is batch-resumable (gpt_train_qa.py writes
# <lang>.qa_progress.jsonl), so a restart never loses completed batches.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

# Wait for the full raw file (defensive; generation completes first).
while [ "$(wc -l < data/training/v1/en.raw.jsonl 2>/dev/null || echo 0)" -lt 3000 ]; do
  sleep 120
done

# Training data is study material: run the review (and top-up
# regeneration) on the free in-house gateway instead of Codex quota.
export EEA_QI_GEN_MODEL="EEA/Inhouse-LLM/gemma-4-31B-it"
export EEA_QI_QA_MODEL="EEA/Inhouse-LLM/gemma-4-31B-it"
uv run python scripts/gpt_train_qa.py en
