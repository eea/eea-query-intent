# EEA Query Intent

A small multilingual NLP service that decides whether a website search query
is **AI-eligible** (a natural-language question, an exploratory request, or a
factual claim) or **no-AI** (keyword/document retrieval, or anything
uncertain). It replaces the rule-based `classifyQueryIntent` in
`volto-searchlib`: only AI-eligible queries may trigger the AI-summary LLM
call. **Everything fails closed** — below threshold, error, timeout, URL, or
empty input means no AI.

- 28 supported languages (24 EU official + Icelandic, Turkish, Bokmål, Nynorsk)
- Model: **setfit-v1** — SetFit linear head on `intfloat/multilingual-e5-small`
- Threshold: **0.95** (sum of the three eligible-label probabilities)
- CPU-only; ~15 ms per query on modern hardware

## Repository layout

```
src/eea_query_intent/     the service (FastAPI): POST /v1/classify, GET /health
tests/                    service tests (27)
configs/setfit-v1.yaml    the exact recipe that produced the production model
data/MANIFEST.json        hashes of all frozen artifacts (do not modify)
data/training/v1/         per-language source corpora (28 languages)
data/banks/v1-short/      170-row short-phrase bank per language (stratum E)
data/pilot/v1/            the frozen training mix + calibration that trained setfit-v1
data/acceptance/v1|v2/    the two frozen exams (v2 = v1 + short-row slice)
data/hygiene/.../         the hygiene adjudication (drop_decisions.jsonl)
docs/adr/                 accepted design decisions (0001 bake-off, 0002 binary gate, 0003 acceptance)
docs/benchmark-results/   committed final benchmark (setfit-v1.json)
docs/DATA_PROVENANCE.md   model card / publication provenance (read before the HF push)
scripts/                  the training methodology (see below)
models/setfit/            the PRODUCTION model (gitignored, ~500 MB)
Dockerfile                container image; fetches the model from Hugging Face at build time
```

## The service

```
uv run python -m eea_query_intent.service     # http://127.0.0.1:8100
```

| endpoint | behavior |
|---|---|
| `POST /v1/classify` `{"query": ...}` | `{intent, eligible, confidence, eligible_probability, abstained, reason, model_version, latency_ms}` |
| `GET /health` | model version, threshold, uptime |

Environment: `EEA_QI_MODEL_PATH` (default `models/setfit`), `EEA_QI_DEVICE`
(`cpu`/`mps`/`cuda`), `EEA_QI_HOST` (default `127.0.0.1`; containers set
`0.0.0.0`), `EEA_QI_PORT` (8100), `EEA_QI_MAX_WORDS` (20),
`EEA_QI_ABSTAIN_THRESHOLD` (overrides the manifest).

Any classifier error or timeout returns HTTP 503 so clients fail closed.
The model must receive casefolded input — the service casefolds every query.

### Running it on this machine (launchd)

The service runs as launchd job `com.razvan.eeaki-service`:
`launchctl submit -l com.razvan.eeaki-service -- /bin/bash /Users/razvan/Work/eea-query-intent/scripts/run_service.sh`
(stdout `/tmp/qi-service.log`, stderr `/tmp/qi-service_err.log`, keepalive).
Verify with `launchctl list | grep eeaki` and `curl 127.0.0.1:8100/health`.
After a full machine shutdown the job may have vanished — resubmit it.
Never edit `run_service.sh` while the service is running.

### Deploying (Rancher)

1. Model and dataset are published: `eeahugs/query-intent-setfit-v1`
   (model, MIT) and `eeahugs/query-intent-setfit-v1-train` (training data,
   CC-BY-NC-4.0 — NLLB/opus-mt provenance disclosed in the model card).
2. Build the image (repo + revision already pinned in the Dockerfile to push
   commit `ba111788`):
   `docker build -t eea-query-intent:1.0.0 .`
   Build on the x86-64 target (the local Mac build is arm64 — do not push an
   arm64 image to x86 pods). Override `HF_MODEL_REPO`/`HF_MODEL_REVISION`
   only for a deliberately new model version.
3. Pod sizing: **2 vCPU / 2 GiB** (measured container RSS ~765 MB; the 1 GiB
   tier is too tight). Probes on `GET /health`: startup ~30 s initial delay,
   10 s period, ~12 failures allowed; same path as readiness (cold start
   loads ~450 MB of weights, healthy in < 60 s).
4. Point the Volto app's `QUERY_INTENT_SERVICE_URL` at the service origin
   (no path — the `/_qi` middleware appends `/v1/classify`). Optional:
   `QUERY_INTENT_TIMEOUT_MS` (default 2000). No frontend changes needed;
   the classifier contract is unchanged. Until the URL is set, AI summaries
   fail closed (search works, no summary).
5. After deployment: run a quick pod latency benchmark (spec: p95 <= 150 ms
   at concurrency 1) — the local M1 numbers (p50 37 ms / p95 76 ms) do not
   transfer to small x86 vCPUs.

## Training methodology (scripts/)

| script | role |
|---|---|
| `train_setfit.py` | trains a SetFit model (`--seed`, `--backbone`, `--binary`); emits calibration predictions |
| `v1_train_eval.sh` | the 3-seed final training loop (families d1 / binary / backbone) |
| `v1_select_seed.py` | ranks seeds on the calibration set; writes `selection.json` |
| `choose_final_threshold.py` | derives the abstain threshold from calibration predictions |
| `sweep_acceptance.py` | threshold sweep / re-gating of raw predictions (`regate`) |
| `predict_acceptance.py` | runs a model over an exam or calibration set (raw predictions) |
| `evaluate_per_language.py`, `validate_acceptance.py` | per-language metrics; exam integrity checks |
| `v1_make_mix.py` | rebuilds the training mix + calibration + exam v2 from source corpora, bank, pools, and drop decisions (deterministic, sha-verified) |
| `v1_author_pools.py`, `pilot_translate_nllb.py` | authored the exam/calibration concept pools; NLLB/opus translation engine |
| `gpt_train_gen.py`, `gpt_train_qa.py` | GPT-native corpus generation + QA (GPT phase paused; see ADR 0003) |
| `gpt_acceptance_gen.py`, `gpt_acceptance_qa.py` | GPT exam generation + QA |
| `train_one_lang.sh`, `probe_lang_confidence.py` | per-language training driver; the language-routing confidence probe |
| `hygiene_scan.py` | deterministic data-hygiene sweep (charset/length/dupes/label conflicts) |
| `smoke_service.py`, `run_service.sh` | service smoke test; launchd entry point |

Rebuilding the production mix after a data edit:
edit the source corpora/banks, run `uv run python scripts/v1_make_mix.py`,
verify the three output SHA-256s changed only where intended
(`data/MANIFEST.json` records the frozen values), then retrain per
`configs/setfit-v1.yaml` and re-run an exam.

## Acceptance criteria (ADR 0003)

The production gate is the **binary routing gate**: worst-language no-AI
false-positive rate <= 1% at the chosen threshold, with high eligible
precision. Five-class macro-F1 is diagnostic only. setfit-v1 at 0.95 scores
worst-language FP 0.5% (Maltese) and average eligible recall 0.859 on the
frozen exam v2 — see `docs/benchmark-results/setfit-v1.json`.

Known accepted scope: out-of-domain general-knowledge queries and some short
English fact-lookups abstain (safe failure: no summary, never a wrong one).
Maltese/Irish/Icelandic remain the weakest languages (gap accepted, ADR 0003).

## Tests

```
CI=true uv run pytest          # 27 tests
uv run ruff check .
```
