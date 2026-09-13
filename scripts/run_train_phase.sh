#!/bin/bash
# 27-language training-corpus phase (relaunch_all.sh stage 6).
# Runs train_one_lang.sh (gen + QA with probe-routed models) per language
# at 4-way parallelism. Per-language logs: /tmp/trn_1_<lang>.log.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

# guard: valid 27-language list, rebuild if missing/broken
if [ ! -s .pipeline/trn_langs.txt ] || \
   [ "$(tr -d ' ' < .pipeline/trn_langs.txt | wc -c)" -lt 54 ]; then
  uv run python -c "from eea_query_intent.languages import SUPPORTED_LANGUAGE_CODES; print(' '.join(sorted(SUPPORTED_LANGUAGE_CODES - {'en'})))" > .pipeline/trn_langs.txt.tmp 2>/dev/null
  if [ "$(tr -d ' ' < .pipeline/trn_langs.txt.tmp 2>/dev/null | wc -c)" -ge 54 ]; then
    mv .pipeline/trn_langs.txt.tmp .pipeline/trn_langs.txt
  else
    rm -f .pipeline/trn_langs.txt.tmp
    echo "ERROR: cannot build language list"
    exit 1
  fi
fi

if [ ! -f .pipeline/train_routing.sh ]; then
  echo "WARNING: no routing env - all languages run on the in-house default"
fi

echo "train phase starting $(date)"
# -L 1: one language per line per invocation (BSD xargs -I does not split on spaces)
cat .pipeline/trn_langs.txt | tr ' ' '\n' | xargs -L 1 -P 4 -I {} bash scripts/train_one_lang.sh {}
echo "train phase done $(date)"
touch .pipeline/trn_phase_done.flag
