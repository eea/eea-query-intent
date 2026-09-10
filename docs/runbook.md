# Runbook — eea-query-intent service

How to run the classifier service locally and query it.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) on PATH
- The model artifacts (already trained and checked into `models/`
  locally; they are git-ignored — retrain with the commands at the
  bottom if missing)

## One-time setup

```bash
cd ~/Work/eea-query-intent
uv sync --all-groups
```

This installs the `ml` group (fasttext-wheel, setfit, onnxruntime,
numpy<2) and the `service` group (fastapi, uvicorn).

## Run the service

```bash
# default: the SetFit model (recommended candidate), port 8100
uv run python -m eea_query_intent.service

# or: the fastText candidate
EEA_QI_MODEL_TYPE=fasttext uv run python -m eea_query_intent.service
```

The service binds `127.0.0.1:8100` and loads the model from
`models/setfit` (or `models/fasttext`). First request after boot pays a
~0.5 s warm-up; subsequent requests run in ~10–30 ms (SetFit on this
Mac; expect a few tens of ms on a small x86-64 CPU — the 150 ms p95
target is the acceptance bar).

Useful environment variables (all optional):

| variable | default | meaning |
|---|---|---|
| `EEA_QI_MODEL_TYPE` | `setfit` | `setfit` or `fasttext` |
| `EEA_QI_MODEL_PATH` | `models/<type>` | model directory |
| `EEA_QI_ABSTAIN_THRESHOLD` | manifest value (0.86 setfit / 0.98 fasttext) | queries below this eligible-probability abstain → no AI |
| `EEA_QI_DEVICE` | `cpu` | `cpu`, `mps`, or `cuda` |
| `EEA_QI_MAX_WORDS` | `20` | overlong queries fail closed without a model call |
| `EEA_QI_PORT` | `8100` | bind port |

## Query it

Health:

```bash
curl -s http://127.0.0.1:8100/health
# {"status":"ok","model_type":"setfit","model_version":"setfit-v1",
#  "abstain_threshold":0.86,"max_words":20,"uptime_seconds":12.3}
```

Classify (works in any of the 28 supported languages):

```bash
curl -s -X POST http://127.0.0.1:8100/v1/classify \
  -H 'Content-Type: application/json' \
  -d '{"query": "What are the main sources of air pollution in Europe?"}'
```

Example responses:

```json
{"intent":"question","eligible":true,"confidence":1.0,
 "eligible_probability":1.0,"abstained":false,"reason":"classified",
 "model_version":"setfit-v1","latency_ms":14.2}
```

```json
{"intent":"unknown","eligible":false,"confidence":0.98,
 "eligible_probability":0.02,"abstained":true,"reason":"below_threshold",
 "model_version":"setfit-v1","latency_ms":12.1}
```

```json
{"intent":"unknown","eligible":false,"confidence":1.0,
 "eligible_probability":0.0,"abstained":false,"reason":"too_long",
 "model_version":"setfit-v1","latency_ms":0.05}
```

Reading the contract:

- `eligible: true` → the search page may generate an AI summary.
- `eligible: false` for any reason → no AI summary, regular search only.
- `reason`: `empty` / `too_long` (deterministic local guard, no model
  call), `below_threshold` (model ran but abstained), `classified`
  (model decision applied).
- `intent` is informational (question / exploratory / claim /
  retrieval / unknown); routing only consumes `eligible`.
- A 503 `{"error": "classifier unavailable"}` means the model failed —
  clients must treat it as `eligible: false` (fail closed).

## Smoke test

```bash
uv run python scripts/smoke_service.py
# hits /health and 17 hand-picked multilingual queries
```

## What to do before production

In priority order (details in `docs/BENCHMARK_RESULTS.md` and
`docs/benchmark-spec.md`):

1. Replace `policy_reviewed` synthetic rows in validation/calibration/
   test with `native_reviewed` data — this is the current
   acceptance blocker.
2. Re-run `scripts/calibrate.py` on the reviewed data; the threshold
   (0.86) will move.
3. Export the SetFit artifact to ONNX + dynamic INT8 (~466 MB fp32 →
   ~120 MB) and re-verify accuracy and calibration on the INT8
   artifact.
4. Measure latency/RSS on the real small-CPU box (Linux x86-64 AVX2),
   not this ARM dev machine.

## Retrain from scratch (if `models/` is missing)

```bash
uv run python scripts/generate_multilingual.py   # regenerate dataset v1
uv run python scripts/train_fasttext.py
uv run python scripts/train_setfit.py
uv run python scripts/calibrate.py fasttext
uv run python scripts/calibrate.py setfit
uv run python scripts/apply_threshold.py fasttext
uv run python scripts/apply_threshold.py setfit
```

SetFit training takes a few minutes on Apple Silicon MPS (2 epochs over
4,693 train rows).
