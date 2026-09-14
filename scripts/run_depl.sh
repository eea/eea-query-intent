#!/bin/bash
# Sequential de + pl training workers, target of the first-class
# launchd job com.razvan.eeaki-depl. Both corpora are in-house routed
# and resumable (raw file + qa_progress checkpoints), so a job or
# machine restart loses at most one in-flight batch.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"
export GPT_FALLBACK=0

for lang in de pl; do
  if [ -f "data/training/v1/${lang}.jsonl" ] && \
     [ "$(wc -l < "data/training/v1/${lang}.jsonl")" -ge 2800 ]; then
    echo "${lang}: already complete, skipping"
    continue
  fi
  echo "=== ${lang} started $(date) ==="
  bash scripts/train_one_lang.sh "$lang"
  echo "=== ${lang} exited code $? at $(date) ==="
done
echo "depl run finished $(date)"
