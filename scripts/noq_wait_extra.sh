#!/bin/bash
# Waits for the main short-bank translation (pid $1) to finish, then runs
# the second translation pass for the extra exploratory/claim rows, and
# touches the sentinel the orchestrator waits on.
set -uo pipefail
cd /Users/razvan/Work/eea-query-intent
exec >> /tmp/noq_extra_translate.log 2>&1

TRANSLATE_PID="$1"
echo "=== extra chain started $(date) (waiting on main pid ${TRANSLATE_PID}) ==="

while kill -0 "${TRANSLATE_PID}" 2>/dev/null; do
  sleep 30
done
if grep -q "^ERROR" /tmp/noq_translate.log; then
  echo "main translation errored - aborting"
  exit 1
fi

uv run --with sentencepiece python scripts/noq_translate_extra.py || exit 1

touch .pipeline/noq_translate_done.flag
echo "=== extra chain DONE $(date) ==="
