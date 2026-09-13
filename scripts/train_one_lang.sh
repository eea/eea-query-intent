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
GPTM="openai-codex/gpt-5.6-luna"
# GPT_FALLBACK=0 pauses every GPT fallback (the user needs the Codex quota
# for their own work); in-house failures then exit FAILED instead of
# silently consuming GPT calls.
GPT_FALLBACK="${GPT_FALLBACK:-1}"
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

# Persistent GPT pause: while .pipeline/gpt_paused exists (the user needs
# the Codex quota for their own work), GPT-routed languages are skipped
# without any GPT call and the in-house GPT fallback is disabled. Removing
# the file resumes the GPT languages on the next relaunch cycle.
if [ -f .pipeline/gpt_paused ]; then
  if [ "$GEN_M" = "$GPTM" ] || [ "$QA_M" = "$GPTM" ]; then
    echo "${LANG_}: GPT paused (.pipeline/gpt_paused) - skipping, no GPT calls"
    exit 3
  fi
  GPT_FALLBACK=0
fi

# run_step <script> <model>: run one pipeline step; the step is resumable,
# so a retry after a failure continues where it stopped.
run_step() {
  export EEA_QI_GEN_MODEL="$2" EEA_QI_QA_MODEL="$2"
  uv run python "scripts/$1" "$LANG_"
}

echo "${LANG_} start $(date) gen_model=${GEN_M}"
if ! run_step gpt_train_gen.py "$GEN_M"; then
  if [ "$GEN_M" != "$GPTM" ] && [ "$GPT_FALLBACK" = "1" ]; then
    # In-house failed (truncated JSON, gateway hiccup...): the model is
    # "not sure" about this language, so fall back to GPT per the routing
    # rule. The partial raw file makes the retry continue per intent.
    echo "${LANG_}: in-house gen failed - falling back to GPT"
    if ! run_step gpt_train_gen.py "$GPTM"; then
      echo "${LANG_}: gen FAILED (GPT fallback too)"; exit 1
    fi
    GEN_M="$GPTM"
  else
    echo "${LANG_}: gen FAILED"; exit 1
  fi
fi
echo "${LANG_} qa $(date) qa_model=${QA_M}"
if ! run_step gpt_train_qa.py "$QA_M"; then
  if [ "$QA_M" != "$GPTM" ] && [ "$GPT_FALLBACK" = "1" ]; then
    echo "${LANG_}: in-house QA failed - falling back to GPT"
    if ! run_step gpt_train_qa.py "$GPTM"; then
      echo "${LANG_}: qa FAILED (GPT fallback too)"; exit 1
    fi
  else
    echo "${LANG_}: qa FAILED"; exit 1
  fi
fi
echo "${LANG_} complete $(date) ($(wc -l < "data/training/v1/${LANG_}.jsonl") rows)"
