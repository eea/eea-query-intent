#!/bin/bash
# Long-running classifier service, target of the first-class launchd
# job com.razvan.eeaki-service (KeepAlive restarts it on crash).
cd /Users/razvan/Work/eea-query-intent
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"
exec env EEA_QI_DEVICE=mps uv run python -m eea_query_intent.service
