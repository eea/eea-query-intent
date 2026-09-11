# EEA Query Intent

Tooling to replace the rule-based (`classifyQueryIntent`) search-query
classifier in `volto-searchlib` with a small multilingual NLP service that runs
on a CPU-only server.

The service decides, for each advanced-search query in any supported EEA
language, whether the query is **AI-eligible** (a natural-language question,
exploratory request, or factual claim) or **no-AI** (keyword/document
retrieval or anything uncertain). Only AI-eligible queries may trigger the AI
summary LLM call. Everything fails closed: no match, low confidence, a URL, or
any error means **no AI**.

## How it works

1. **Deterministic policy guards** run first (local, synchronous, no model):
   empty input, over-20-word input, and pasted URLs fail closed.
2. **SetFit model** (frozen multilingual MiniLM-L12 backbone + a small linear
   head) produces the five intent probabilities.
3. **Abstention threshold** (calibrated, default `0.98`): if the summed
   AI-eligible probability is below it, the query abstains (no AI).

## Data

- **English anchor bank**: 460 hand-authored queries
  (`data/english/expanded_v1.jsonl`, source in `scripts/english_bank_data.py`),
  split 413 train / 47 calibration.
- **Per-row translations**: every bank row is translated into all 27 other
  supported languages. The source-of-truth modules are
  `scripts/translations/<lang>.py`; they are serialized to
  `data/translations/<lang>.jsonl` by `scripts/make_translations.py`.
- **Review status**: rows are marked `llm_reviewed` — GPT-5.6 Sol (via the pi
  CLI) is the translation reviewer for this project. The acceptance gate
  accepts `native_reviewed` **or** `llm_reviewed`.
- **Training set**: `scripts/make_expanded_v2.py` folds the English set
  (`data/english_only/`) and every translation into
  `data/expanded_v2/{train,calibration}.jsonl`. Each translation is assigned
  its English source row's split so calibration rows never leak into training.

## Current status

- **SetFit-only FastAPI service** (`src/eea_query_intent/service.py`):
  `POST /v1/classify`, `GET /health`, 503 fail-closed on adapter errors. The
  adapter reads and validates the label order from the model manifest.
- **All 28 supported languages** have hand-anchored, GPT-QA'd per-row
  translations. Final model `setfit-v3` (11,772 train rows) passes the
  no-AI false-routing gate for **25/25** held-out test languages at the
  `0.98` threshold.
- **URL guard**: pasted URLs (`http://`, `https://`, `www.`) fail closed
  before the model, so they never trigger an AI summary.

**Honest limitations**

- The held-out test split has only 29 rows per language, so "0 false routes
  in 9 no-AI rows" is not strong statistical evidence of a ≤1% rate. The
  acceptance gates (300 reviewed examples per language, ≥98% eligible
  precision, ≥95% macro-F1) are deliberately still failing on **data volume**.
- Translations of one 460-row English bank add language coverage but limited
  query diversity (no typos, navigation searches, real production distribution).
- The `0.98` threshold is conservative by design (fail-closed); it trades some
  eligible-query recall for safety. Short topical phrases remain the hardest
  no-AI/eligible boundary.

## Next steps

- Native-speaker-reviewed data at the 300-per-language bar (the acceptance
  blocker) — retrain and re-tune the threshold once it exists.
- ONNX + dynamic INT8 export of the SetFit artifact (~466 MB → ~120 MB) with
  accuracy and calibration re-verified on the INT8 artifact.
- Latency/RSS/cold-start measurement on the target small-CPU box.
- Frontend swap in `volto-searchlib` (async intent fetch, query-keyed state,
  cancellation, timeout/fail-closed, E2E network assertions).

## Language scope

The 24 official EU languages plus Icelandic, Turkish, and Norwegian (both
written standards, `nb` and `nn`). See
`src/eea_query_intent/languages.py` for the canonical list.

## Quickstart

```sh
uv sync --all-groups
uv run pytest
uv run eea-query-intent data validate data/expanded_v2/train.jsonl

# run the classifier service (SetFit, port 8100)
uv run python -m eea_query_intent.service

# query it
curl -s -X POST http://127.0.0.1:8100/v1/classify \
  -H 'Content-Type: application/json' \
  -d '{"query": "What are the main sources of air pollution in Europe?"}'
```

Full instructions, environment variables, and retraining commands are in
[docs/runbook.md](docs/runbook.md).

## Layout

```
src/eea_query_intent/
  contracts.py     shared intent/eligibility contract
  languages.py     supported language codes (28)
  policy.py        local fail-closed guards (empty, max words, URL)
  dataset.py       JSONL dataset format, validation, leakage rules
  metrics.py       per-language asymmetric evaluation + acceptance gates
  service.py       FastAPI service (SetFit adapter)
  cli.py           `data validate` / `evaluate` commands
data/english/      hand-authored English anchor bank
data/english_only/ frozen English training/calibration set
data/translations/ per-language QA'd translation JSONL (27 languages)
data/expanded_v2/  final assembled train/calibration set
data/multilingual/ held-out test split (regression set)
models/            trained model artifacts (git-ignored)
docs/              annotation spec, benchmark spec, results, runbook, ADRs
scripts/           data generation, translation, training, calibration
```

## Key documents

- [docs/annotation-spec.md](docs/annotation-spec.md) — label definitions and
  dataset rules
- [docs/benchmark-spec.md](docs/benchmark-spec.md) — hardware targets,
  performance gates, candidate models
- [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) — bake-off evidence
  and honest failure analysis
- [docs/runbook.md](docs/runbook.md) — run and query the service
- [docs/adr/](docs/adr/) — decisions
