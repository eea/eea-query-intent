# Runbook — eea-query-intent service

How to run the classifier service locally and query it.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) on PATH
- The model artifact (already trained into `models/setfit`; it is
  git-ignored — retrain with the commands at the bottom if missing)

## One-time setup

```bash
cd ~/Work/eea-query-intent
uv sync --all-groups
```

This installs the `ml` group (setfit, onnxruntime, numpy<2) and the `service`
group (fastapi, uvicorn).

## Run the service

```bash
# the SetFit model, port 8100
uv run python -m eea_query_intent.service
```

The service binds `127.0.0.1:8100` and loads the model from `models/setfit`.
First request after boot pays a ~0.5 s warm-up; subsequent requests run in
~10–30 ms (SetFit on this Mac; expect a few tens of ms on a small x86-64 CPU —
the 150 ms p95 target is the acceptance bar).

Useful environment variables (all optional):

| variable | default | meaning |
|---|---|---|
| `EEA_QI_MODEL_PATH` | `models/setfit` | model directory |
| `EEA_QI_ABSTAIN_THRESHOLD` | manifest value (`0.98`) | queries below this eligible-probability abstain → no AI |
| `EEA_QI_DEVICE` | `cpu` | `cpu`, `mps`, or `cuda` |
| `EEA_QI_MAX_WORDS` | `20` | overlong queries fail closed without a model call |
| `EEA_QI_PORT` | `8100` | bind port |

## Query it

Health:

```bash
curl -s http://127.0.0.1:8100/health
# {"status":"ok","model_type":"setfit","model_version":"setfit-v3",
#  "abstain_threshold":0.98,"max_words":20,"uptime_seconds":12.3}
```

Classify (works in any of the 28 supported languages):

```bash
curl -s -X POST http://127.0.0.1:8100/v1/classify \
  -H 'Content-Type: application/json' \
  -d '{"query": "What are the main sources of air pollution in Europe?"}'
```

Example responses:

```json
{"intent":"question","eligible":true,"confidence":0.99,
 "eligible_probability":0.99,"abstained":false,"reason":"classified",
 "model_version":"setfit-v3","latency_ms":14.2}
```

```json
{"intent":"unknown","eligible":false,"confidence":0.98,
 "eligible_probability":0.02,"abstained":true,"reason":"below_threshold",
 "model_version":"setfit-v3","latency_ms":12.1}
```

```json
{"intent":"unknown","eligible":false,"confidence":1.0,
 "eligible_probability":0.0,"abstained":false,"reason":"url",
 "model_version":"setfit-v3","latency_ms":0.05}
```

Reading the contract:

- `eligible: true` → the search page may generate an AI summary.
- `eligible: false` for any reason → no AI summary, regular search only.
- `reason`: `empty` / `too_long` / `url` (deterministic local guard, no model
  call), `below_threshold` (model ran but abstained), `classified`
  (model decision applied).
- `intent` is informational (question / exploratory / claim /
  retrieval / unknown); routing only consumes `eligible`.
- A 503 `{"error": "classifier unavailable"}` means the model failed —
  clients must treat it as `eligible: false` (fail closed).

## Smoke test

```bash
uv run python scripts/smoke_service.py
# hits /health and a set of multilingual queries with expected eligibility
```

## What to do before production

In priority order (details in `docs/BENCHMARK_RESULTS.md` and
`docs/benchmark-spec.md`):

1. Replace the GPT-QA'd (`llm_reviewed`) synthetic rows in validation/
   calibration/test with native-speaker-reviewed data at the 300-per-language
   bar — this is the current acceptance blocker.
2. Re-run `scripts/calibrate.py setfit` on the reviewed data; the threshold
   (`0.98`) will move.
3. Export the SetFit artifact to ONNX + dynamic INT8 (~466 MB fp32 → ~120 MB)
   and re-verify accuracy and calibration on the INT8 artifact.
4. Measure latency/RSS on the real small-CPU box (Linux x86-64 AVX2),
   not this ARM dev machine.

## Retrain from scratch (if `models/` is missing)

```bash
# assemble the final train/calibration set (English + all 27 translations)
uv run python scripts/make_expanded_v2.py
# train the SetFit head and emit test + calibration predictions
uv run python scripts/train_setfit.py \
  --train-file data/expanded_v2/train.jsonl \
  --calibration-file data/expanded_v2/calibration.jsonl \
  --test-file data/multilingual/v1/test.jsonl \
  --model-dir models/setfit --model-version setfit-v3
# pick the abstention threshold on the calibration split
uv run python scripts/calibrate.py setfit \
  --model-dir models/setfit --calibration-file data/expanded_v2/calibration.jsonl
```

SetFit training takes a few minutes on Apple Silicon MPS (2 epochs over the
train split).
