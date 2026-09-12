#!/bin/bash
# Acceptance-generation phase: run gpt_acceptance_gen for all 27 non-English
# languages at 4-way parallelism, then mark the phase done. Each language is
# crash-resilient (the gen script resumes per intent).
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

# guard: valid 27-language list, rebuild if missing/broken
if [ ! -s .pipeline/acc_langs.txt ] || [ "$(tr -d ' ' < .pipeline/acc_langs.txt | wc -c)" -lt 54 ]; then
  uv run python -c "from eea_query_intent.languages import SUPPORTED_LANGUAGE_CODES; print(' '.join(sorted(SUPPORTED_LANGUAGE_CODES - {'en'})))" > .pipeline/acc_langs.txt.tmp 2>/dev/null
  if [ "$(tr -d ' ' < .pipeline/acc_langs.txt.tmp 2>/dev/null | wc -c)" -ge 54 ]; then
    mv .pipeline/acc_langs.txt.tmp .pipeline/acc_langs.txt
  else
    rm -f .pipeline/acc_langs.txt.tmp
    echo "ERROR: cannot build language list"
    exit 1
  fi
fi

echo "gen phase starting $(date)"
# -L 1: one language per line per invocation (BSD xargs -I does not split on spaces)
cat .pipeline/acc_langs.txt | tr ' ' '\n' | xargs -L 1 -P 4 -I {} bash scripts/gen_one_lang.sh {}
echo "gen phase done $(date)"
touch .pipeline/acc_gen_done.flag
