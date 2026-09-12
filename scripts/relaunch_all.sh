#!/bin/bash
# Idempotent self-healing relauncher for the eea-query-intent data pipeline.
# Designed to be run every few minutes by a launchd agent: it detects the
# real state on disk and via pid files, and (re)launches whatever is missing.
# Safe to run any number of times.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p .pipeline

# keep the system awake while the pipeline is pending
if [ ! -f .pipeline/en_validation_done.flag ] && ! pgrep -x caffeinate > /dev/null; then
  nohup caffeinate -s > /dev/null 2>&1 &
fi

pid_alive() {
  local pidfile="$1"
  [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile" 2>/dev/null)" 2>/dev/null
}

# 1. acceptance generation: exactly one orchestrator.
# Completion is data-derived: all 27 raw files present with 825 rows.
# Gen workers are resumable, so orphaned ones are killed before relaunching.
lang_list_ok() {
  local file="${1:-.pipeline/acc_langs.txt}"
  [ -s "$file" ] || return 1
  [ "$(tr -d ' ' < "$file" | wc -c)" -ge 54 ]
}

refresh_lang_list() {
  uv run python -c "from eea_query_intent.languages import SUPPORTED_LANGUAGE_CODES; print(' '.join(sorted(SUPPORTED_LANGUAGE_CODES - {'en'})))" > .pipeline/acc_langs.txt.tmp 2>/dev/null
  if lang_list_ok .pipeline/acc_langs.txt.tmp; then
    mv .pipeline/acc_langs.txt.tmp .pipeline/acc_langs.txt
  else
    rm -f .pipeline/acc_langs.txt.tmp
    return 1
  fi
}

all_raws() {
  lang_list_ok || return 1
  local missing=0 f
  for f in $(cat .pipeline/acc_langs.txt); do
    [ -f "data/acceptance/v1/$f.raw.jsonl" ] || { missing=1; break; }
    [ "$(wc -l < "data/acceptance/v1/$f.raw.jsonl")" -ge 825 ] || { missing=1; break; }
  done
  return $missing
}

if all_raws; then
  echo "acceptance generation complete (all 27 raw files)"
elif pid_alive .pipeline/gen.pid; then
  echo "acceptance generation orchestrator running (pid $(cat .pipeline/gen.pid))"
else
  pkill -f "xargs -P 4 -I .*gpt_acceptance_gen" 2>/dev/null
  pkill -f "\.venv/bin/python3 scripts/gpt_acceptance_gen.py" 2>/dev/null
  sleep 2
  refresh_lang_list || { echo "ERROR: cannot build language list"; }
  nohup bash -c 'cat .pipeline/acc_langs.txt | xargs -P 4 -I {} bash -c "uv run python scripts/gpt_acceptance_gen.py {} > /tmp/acc_gen_{}.log 2>&1"; touch .pipeline/acc_gen_done.flag' \
    > /tmp/acc_orchestrator.log 2>&1 &
  echo $! > .pipeline/gen.pid
  echo "launched acceptance generation orchestrator (pid $!)"
fi

# 2. acceptance QA phase (idempotent per language). Completion is
# data-derived: all 27 final <lang>.jsonl present. QA workers are NOT
# resumable, so if orphaned QA workers are alive we do not launch a second
# phase (they finish their languages; the next cycle launches for the rest).
all_finals() {
  lang_list_ok || return 1
  local f
  for f in $(cat .pipeline/acc_langs.txt); do
    [ -f "data/acceptance/v1/$f.jsonl" ] || return 1
  done
  return 0
}

if all_finals; then
  echo "acceptance QA phase complete (all 27 final files)"
elif pid_alive .pipeline/qa.pid; then
  echo "acceptance QA phase running (pid $(cat .pipeline/qa.pid))"
elif pgrep -f "\.venv/bin/python3 scripts/gpt_acceptance_qa.py" > /dev/null; then
  echo "acceptance QA workers alive, phase shell missing - letting them finish"
else
  nohup bash scripts/run_acceptance_qa_phase.sh > /tmp/acc_qa_phase.log 2>&1 &
  echo $! > .pipeline/qa.pid
  echo "launched acceptance QA phase (pid $!)"
fi

# 3. English training generation (single instance)
if [ -f data/training/v1/en.raw.jsonl ] && [ "$(wc -l < data/training/v1/en.raw.jsonl)" -ge 3000 ]; then
  echo "English training generation complete"
elif pid_alive .pipeline/entrgen.pid; then
  echo "English training generation running (pid $(cat .pipeline/entrgen.pid))"
else
  pkill -f "gpt_train_gen.py en" 2>/dev/null
  sleep 1
  nohup uv run python scripts/gpt_train_gen.py en > /tmp/trn_gen_en.log 2>&1 &
  echo $! > .pipeline/entrgen.pid
  echo "launched English training generation (pid $!)"
fi

# 4. English training QA (single instance: real worker OR pending watcher)
if [ -f data/training/v1/en.jsonl ]; then
  echo "English training QA complete"
elif pid_alive .pipeline/enqa.pid; then
  echo "English training QA running (pid $(cat .pipeline/enqa.pid))"
else
  pkill -f "gpt_train_qa.py en" 2>/dev/null
  sleep 1
  nohup bash -c '
    cd /Users/razvan/Work/eea-query-intent
    while [ "$(wc -l < data/training/v1/en.raw.jsonl 2>/dev/null || echo 0)" -lt 3000 ]; do sleep 120; done
    uv run python scripts/gpt_train_qa.py en > /tmp/trn_qa_en.log 2>&1
  ' > /tmp/trn_watch_en.log 2>&1 &
  echo $! > .pipeline/enqa.pid
  echo "launched English training QA watcher (pid $!)"
fi

# 5. English validation watcher (single instance)
if [ -f .pipeline/en_validation_done.flag ]; then
  echo "English validation complete"
elif pid_alive .pipeline/enval.pid; then
  echo "English validation watcher running (pid $(cat .pipeline/enval.pid))"
else
  nohup bash scripts/watch_en_validation.sh > /tmp/watch_en_validation.log 2>&1 &
  echo $! > .pipeline/enval.pid
  echo "launched English validation watcher (pid $!)"
fi

echo "relaunch checked $(date)"
