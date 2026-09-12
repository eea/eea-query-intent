#!/bin/bash
# Acceptance QA phase: run the QA pass (gpt_acceptance_qa +
# repair_acceptance_overlaps) per language, 4 at a time. Per-language
# idempotency and completeness checks live in scripts/qa_one_lang.sh, so
# this phase can run in parallel with the generation phase: incomplete
# languages are skipped and picked up by the next phase cycle.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

echo "QA phase starting $(date)"
# guard: refuse to run with a broken language list
if [ ! -s .pipeline/acc_langs.txt ] || [ "$(tr -d ' ' < .pipeline/acc_langs.txt | wc -c)" -lt 54 ]; then
  echo "ERROR: language list missing or broken - exiting for retry"
  exit 1
fi
cat .pipeline/acc_langs.txt | tr ' ' '\n' | xargs -L 1 -P 8 -I {} bash scripts/qa_one_lang.sh {}
touch .pipeline/acc_qa_done.flag
echo "QA phase done $(date)"
