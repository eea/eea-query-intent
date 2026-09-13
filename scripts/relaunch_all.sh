#!/bin/bash
# Idempotent self-healing relauncher for the eea-query-intent data pipeline.
# Designed to be run every few minutes by a launchd agent: it detects the
# real state on disk and via pid files, and (re)launches whatever is missing.
# Safe to run any number of times.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p .pipeline
# launchd runs this with a minimal PATH; make the toolchain reachable
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

# keep the system awake while the pipeline is pending
if [ ! -f .pipeline/en_validation_done.flag ] && ! pgrep -x caffeinate > /dev/null; then
  nohup caffeinate -s > /dev/null 2>&1 &
fi

pid_alive() {
  local pidfile="$1" pattern="${2:-}"
  [ -f "$pidfile" ] || return 1
  local pid
  pid=$(cat "$pidfile" 2>/dev/null)
  kill -0 "$pid" 2>/dev/null || return 1
  # guard against PID reuse: the process must still be the expected one
  [ -n "$pattern" ] || return 0
  ps -p "$pid" -o command= 2>/dev/null | grep -q "$pattern"
}

# 1. acceptance generation: exactly one orchestrator.
# Completion is data-derived: all 27 raw files present with 825 rows.
# Gen workers are resumable, so orphaned ones are killed before relaunching.
lang_list_ok() {
  local file="${1:-.pipeline/acc_langs.txt}"
  [ -s "$file" ] || return 1
  [ "$(tr -d ' ' < "$file" | wc -c)" -ge 54 ]
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
elif pid_alive .pipeline/gen.pid "run_gen_phase" || pgrep -f "run_gen_phase.sh" > /dev/null; then
  echo "acceptance generation phase running"
else
  pkill -f "run_gen_phase.sh" 2>/dev/null
  pkill -f "xargs -P 4 -I {} bash scripts/gen_one_lang.sh" 2>/dev/null
  pkill -f "\.venv/bin/python3 scripts/gpt_acceptance_gen.py" 2>/dev/null
  sleep 2
  nohup bash scripts/run_gen_phase.sh > /tmp/acc_orchestrator.log 2>&1 &
  echo $! > .pipeline/gen.pid
  echo "launched acceptance generation phase (pid $!)"
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
elif pid_alive .pipeline/qa.pid "run_acceptance_qa_phase" \
  || pgrep -f "run_acceptance_qa_phase.sh" > /dev/null \
  || pgrep -f "\.venv/bin/python3 scripts/gpt_acceptance_qa.py" > /dev/null; then
  echo "acceptance QA phase running"
else
  nohup bash scripts/run_acceptance_qa_phase.sh > /tmp/acc_qa_phase.log 2>&1 &
  echo $! > .pipeline/qa.pid
  echo "launched acceptance QA phase (pid $!)"
fi

# 3. English training generation (single instance)
if [ -f data/training/v1/en.raw.jsonl ] && [ "$(wc -l < data/training/v1/en.raw.jsonl)" -ge 3000 ]; then
  echo "English training generation complete"
elif pid_alive .pipeline/entrgen.pid "gpt_train_gen"; then
  echo "English training generation running (pid $(cat .pipeline/entrgen.pid))"
else
  pkill -f "gpt_train_gen.py en" 2>/dev/null
  sleep 1
  nohup uv run python scripts/gpt_train_gen.py en > /tmp/trn_gen_en.log 2>&1 &
  echo $! > .pipeline/entrgen.pid
  echo "launched English training generation (pid $!)"
fi

# 4. English training QA (single instance: real worker OR pending watcher).
# Completion requires the final file with all 3000 rows, not mere existence.
en_train_done() {
  [ -f data/training/v1/en.jsonl ] && [ "$(wc -l < data/training/v1/en.jsonl)" -ge 3000 ]
}
enqa_job() {
  launchctl list 2>/dev/null | grep -q "com.razvan.eeaki-trnqa"
}
if en_train_done; then
  # launchctl submit creates KEEPALIVE jobs on this macOS: remove the job
  # once the work is done, or launchd restarts the finished worker forever.
  launchctl remove com.razvan.eeaki-trnqa 2>/dev/null
  echo "English training QA complete (3000 rows)"
elif enqa_job; then
  echo "English training QA running (launchd job)"
else
  MATCHES=$(pgrep -f "gpt_train_qa.py en" 2>/dev/null | tr '\n' ' ')
  echo "enqa: restarting at $(date +%T) (live-matches-before-pkill: ${MATCHES:-none})"
  pkill -f "gpt_train_qa.py en" 2>/dev/null
  sleep 1
  launchctl remove com.razvan.eeaki-trnqa 2>/dev/null
  if launchctl submit -l com.razvan.eeaki-trnqa \
      -o /tmp/trn_qa_en.log -e /tmp/trn_qa_en.log \
      -- /bin/bash /Users/razvan/Work/eea-query-intent/scripts/run_en_qa.sh 2>/dev/null; then
    echo "launched English training QA (launchd job com.razvan.eeaki-trnqa)"
  else
    echo "launchctl submit failed - falling back to nohup"
    nohup bash scripts/run_en_qa.sh > /tmp/trn_qa_en.log 2>&1 &
    echo $! > .pipeline/enqa.pid
  fi
fi

# 5. English validation watcher (single instance). The flag is only valid
# if the training file it was measured on was complete.
enval_job() {
  launchctl list 2>/dev/null | grep -q "com.razvan.eeaki-enval"
}
if [ -f .pipeline/en_validation_done.flag ] && en_train_done; then
  echo "English validation complete"
elif [ -f .pipeline/en_validation_done.flag ]; then
  echo "stale validation flag (training file incomplete) - discarding"
  rm -f .pipeline/en_validation_done.flag
  rm -f .pipeline/enval.pid
  launchctl remove com.razvan.eeaki-enval 2>/dev/null
  launchctl submit -l com.razvan.eeaki-enval \
    -o /tmp/watch_en_validation.log -e /tmp/watch_en_validation.log \
    -- /bin/bash /Users/razvan/Work/eea-query-intent/scripts/watch_en_validation.sh \
    || nohup bash scripts/watch_en_validation.sh > /tmp/watch_en_validation.log 2>&1 &
  echo "re-launched English validation watcher"
elif enval_job || pid_alive .pipeline/enval.pid "watch_en_validation"; then
  echo "English validation watcher running"
else
  launchctl remove com.razvan.eeaki-enval 2>/dev/null
  launchctl submit -l com.razvan.eeaki-enval \
    -o /tmp/watch_en_validation.log -e /tmp/watch_en_validation.log \
    -- /bin/bash /Users/razvan/Work/eea-query-intent/scripts/watch_en_validation.sh \
    || nohup bash scripts/watch_en_validation.sh > /tmp/watch_en_validation.log 2>&1 &
  echo "launched English validation watcher"
fi

# 6. 27-language training corpus phase (run_train_phase.sh).
# Completion: all 28 training finals (en + 27) with 3000 rows each.
trn_all_done() {
  [ -f data/training/v1/en.jsonl ] && [ "$(wc -l < data/training/v1/en.jsonl)" -ge 3000 ] || return 1
  [ -f .pipeline/trn_langs.txt ] || return 1
  for lang in $(cat .pipeline/trn_langs.txt); do
    [ -f "data/training/v1/${lang}.jsonl" ] && \
      [ "$(wc -l < "data/training/v1/${lang}.jsonl")" -ge 3000 ] || return 1
  done
}
trn_phase_running() {
  pgrep -f "run_train_phase.sh" > /dev/null 2>&1 || \
    pgrep -f "gpt_train_gen.py" > /dev/null 2>&1 || \
    pgrep -f "gpt_train_qa.py" > /dev/null 2>&1
}
if trn_all_done; then
  echo "training corpus complete (all 28 languages)"
elif trn_phase_running; then
  echo "training corpus phase running"
else
  MATCHES=$( { pgrep -f "gpt_train_gen.py"; pgrep -f "gpt_train_qa.py"; } 2>/dev/null | tr '\n' ' ')
  echo "trn: restarting at $(date +%T) (live-matches-before-pkill: ${MATCHES:-none})"
  pkill -f "gpt_train_gen.py" 2>/dev/null
  pkill -f "gpt_train_qa.py" 2>/dev/null
  sleep 1
  nohup bash scripts/run_train_phase.sh > /tmp/trn_phase_orch.log 2>&1 &
  echo $! > .pipeline/trnphase.pid
  echo "launched training corpus phase (pid $!)"
fi

# 7. Final build chain (run_final_build.sh). Fires when the training
# corpus AND the acceptance exam (all 28 shards incl. en) are complete.
acceptance_all_done() {
  all_finals && [ -f data/acceptance/v1/en.jsonl ]
}
if [ -f .pipeline/final_build_done.flag ]; then
  echo "final build complete"
elif pgrep -f "run_final_build.sh" > /dev/null 2>&1; then
  echo "final build running"
elif trn_all_done && acceptance_all_done; then
  nohup bash scripts/run_final_build.sh > /tmp/final_build_orch.log 2>&1 &
  echo $! > .pipeline/finalbuild.pid
  echo "launched final build (pid $!)"
else
  echo "final build waiting (training or acceptance incomplete)"
fi

echo "relaunch checked $(date)"
