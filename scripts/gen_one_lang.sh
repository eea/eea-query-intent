#!/bin/bash
# One acceptance-gen language: crash-resilient resume handled inside the script.
uv run python scripts/gpt_acceptance_gen.py "$1" >> "/tmp/acc_gen_$1.log" 2>&1
