# EEA Query Intent

Benchmark-first tooling to replace the rule-based (`classifyQueryIntent`)
search-query classifier in `volto-searchlib` with a small multilingual NLP
model that runs on a CPU-only server.

The service decides, for each advanced-search query in any supported EEA
language, whether the query is **AI-eligible** (a natural-language question,
exploratory request, or factual claim) or **no-AI** (keyword/document
retrieval or anything uncertain). Only AI-eligible queries may trigger the
AI summary LLM call.

## Current status

**Slice 1 — benchmark foundation (done):**

- Dataset format and validator with template-leakage protection
- Asymmetric per-language evaluation metrics and acceptance gates
- Local deterministic policy guards (empty / overlong input)
- English seed corpus (118 rows) generated from the shipped frontend policy
- CLI: `data validate`, `evaluate`

**Slice 2 — multilingual dataset v1 (done):**

- 56 synthetic templates × 28 languages + 118-row English seed = 6,793
  rows, split per template (no template leakage across splits)
- Topics: air pollution, biodiversity, climate, water, PFAS, renewables,
  waste, ozone, PM2.5, deforestation, plastics, urban heat, Green Deal,
  air quality

**Slice 3 — first real bake-off (done, prototype quality):**

- Trained two real candidates: quantized fastText (6.7 MB) and SetFit on
  multilingual MiniLM L6 (466 MB fp32, MPS training)
- Calibrated abstention thresholds on the calibration split:
  **SetFit 0.86** (meets the worst-language no-AI ≤1% target on
  calibration), fastText 0.98 (does not meet it)
- Gated test result: SetFit false-routes no-AI in only 1/28 languages
  (Irish, 2/9 sample noise) vs 2/28 for fastText; see
  [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md)
- Production blocker is native-speaker-reviewed validation data, not
  model size — the gates are correctly rejecting the synthetic prototype

**Slice 4 — the service (done):**

- Model-agnostic FastAPI service: `POST /v1/classify`, `GET /health`,
  503 fail-closed on adapter errors, policy guards before model calls,
  calibrated abstention threshold applied
- Smoke-tested live in 8+ languages; warm latency ~10–30 ms, RSS ~312 MiB
- [docs/runbook.md](docs/runbook.md) — how to run and query it yourself

**Next slices (planned):**

- Native-speaker review pipeline for validation/calibration/test rows
  (the acceptance blocker)
- ONNX + dynamic INT8 export of the SetFit artifact (~466 MB → ~120 MB)
  with accuracy and calibration re-verified on the INT8 artifact
- Latency/RSS/cold-start measurement on the target small-CPU box
  (Linux x86-64 AVX2)
- Frontend swap in `volto-searchlib` (async intent fetch, query-keyed
  state, cancellation, E2E network assertions)

## Language scope

The 24 official EU languages plus Icelandic, Turkish, and Norwegian (both
written standards, `nb` and `nn`). See
`src/eea_query_intent/languages.py` for the canonical list.

## Quickstart

```sh
uv sync --all-groups
uv run pytest
uv run eea-query-intent data validate data/multilingual/v1/train.jsonl

# run the classifier service (SetFit candidate, port 8100)
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
  policy.py        local fail-closed guards (empty, max words)
  dataset.py       JSONL dataset format, validation, leakage rules
  metrics.py       per-language asymmetric evaluation + acceptance gates
  service.py       FastAPI service (model-agnostic adapters)
  cli.py           `data validate` / `evaluate` commands
data/seed/         English seed corpus (from the frontend policy corpus)
data/multilingual/ dataset v1 (train/validation/calibration/test)
models/            trained candidate artifacts (git-ignored)
docs/              annotation spec, benchmark spec, results, runbook, ADRs
scripts/           data generation, training, calibration, service smoke test
```

## Key documents

- [docs/annotation-spec.md](docs/annotation-spec.md) — label definitions and
  dataset rules
- [docs/benchmark-spec.md](docs/benchmark-spec.md) — hardware targets,
  performance gates, candidate models
- [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) — first bake-off
  evidence and honest failure analysis
- [docs/runbook.md](docs/runbook.md) — run and query the service
- [docs/adr/](docs/adr/) — decisions
