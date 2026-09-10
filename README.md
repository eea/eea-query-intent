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

**Next slices (planned):**

- Multilingual gold dataset generation + native review pipeline
- Candidate adapters (fastText, mmBERT-small, multilingual MiniLM L6, e5 +
  SetFit, XLM-R ceiling) with a reproducible train/export pipeline
- Hardware benchmark harness on the target CPU (latency, RSS, cold start)
- The FastAPI service itself (kept model-agnostic until the bake-off winner)

## Language scope

The 24 official EU languages plus Icelandic, Turkish, and Norwegian (both
written standards, `nb` and `nn`). See
`src/eea_query_intent/languages.py` for the canonical list.

## Quickstart

```sh
uv sync
uv run pytest
uv run eea-query-intent data validate data/seed/english.jsonl
```

## Layout

```
src/eea_query_intent/
  contracts.py     shared intent/eligibility contract
  languages.py     supported language codes (28)
  policy.py        local fail-closed guards (empty, max words)
  dataset.py       JSONL dataset format, validation, leakage rules
  metrics.py       per-language asymmetric evaluation + acceptance gates
  cli.py           `data validate` / `evaluate` commands
data/seed/         English seed corpus (from the frontend policy corpus)
docs/              annotation spec, benchmark spec, ADRs
scripts/           corpus generation utilities
```

## Key documents

- [docs/annotation-spec.md](docs/annotation-spec.md) — label definitions and
  dataset rules
- [docs/benchmark-spec.md](docs/benchmark-spec.md) — hardware targets,
  performance gates, candidate models
- [docs/adr/](docs/adr/) — decisions
