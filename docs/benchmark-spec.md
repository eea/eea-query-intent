# Benchmark specification

## Objective

Select the smallest model that meets the per-language acceptance gates on
the target hardware. The winner is decided by the **worst language**
(especially Maltese and Irish), not by average accuracy.

## Deployment profile

Preferred target: **1 vCPU, 1 GiB RAM, CPU-only, Linux x86-64 (AVX2),
batch size 1.** Maximum acceptable tier: 2 vCPU / 2 GiB RAM. ARM is a
separate measurement if production turns out ARM.

Performance gates (queries up to 20 words):

| Metric | Gate |
| --- | --- |
| Warm p95 latency, concurrency 1 | ≤ 150 ms |
| Warm p95 latency, concurrency 4 | ≤ 300 ms |
| Client timeout (frontend) | 500 ms, then fail closed |
| Peak RSS under load | goal ≤ 750 MiB; hard limit ≤ 900 MiB (1 GiB tier) |
| Cold start (process → first request) | ≤ 10 s |
| Runtime model downloads | none (artifact baked in) |
| Swap usage | none |

Size must be **measured** (artifact bytes + runtime RSS), never derived from
parameter counts — e.g. mmBERT-small's 256k embedding table dominates its
footprint and may not quantize as expected.

## Accuracy gates (per language, on the native-reviewed test split)

Reported by `eea-query-intent evaluate`; a language fails if any gate fails,
and the model is promoted only when **no** language fails:

| Gate | Default |
| --- | --- |
| Minimum eligible sample count | 300 |
| Minimum no-AI sample count | 300 |
| No-AI false-positive rate (keyword → LLM) | ≤ 1% |
| Eligible precision (AI-eligible → truly eligible) | ≥ 98% |
| Macro-F1 over intent classes present | ≥ 95% |

Additional reported (non-gating in this slice) metrics: per-intent recall for
question/exploratory/claim, abstention rate and coverage, Brier score
(calibration), p50/p95 latency.

The threshold for abstention is chosen on the `calibration` split using the
risk–coverage curve so that the worst-language false-positive rate meets the
gate; it is then frozen before the `test` split is touched. Raw softmax
probabilities are not treated as calibrated.

## Candidates

| Candidate | Role |
| --- | --- |
| fastText supervised (quantized) | Small-CPU baseline; serious winner candidate for short lexical queries |
| `jhu-clsp/mmBERT-small` fine-tuned | Primary transformer candidate: only small encoder with attested coverage of all 28 languages (MIT) |
| `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` (or fine-tuned MiniLM L6) | Speed-oriented transformer baseline (~107 MB INT8) |
| `intfloat/multilingual-e5-small` + SetFit/linear head | Label-efficient embedding baseline (Maltese out of pretraining distribution — must be proven) |
| XLM-R base | Quality ceiling / offline teacher only (279M params, ~1.12 GB FP32; no Maltese in published list) |

Export: ONNX; for transformer candidates run the ONNX Runtime transformer
optimizations + dynamic INT8 quantization, then **re-verify accuracy and
calibration on the INT8 artifact** (quantization shifts logits).

## Reproducibility requirements

- Every run records: model id + revision, library versions, dataset git
  revision, hardware fingerprint, seeds, and the full report JSON.
- Reports land in `reports/<run-id>/` (git-ignored).
- Training and export are separate steps; the service consumes a pinned
  artifact (hash-checked), never a live Hugging Face download.

## Failure cascade (if no single compact model passes)

1. Quantized fastText answers high-confidence cases.
2. The compact transformer answers the uncertain remainder.
3. Anything still uncertain abstains (no AI).

## Out of scope (this phase)

The FastAPI service, frontend integration, and any language-detection step
(short queries make LID unreliable; the model must be directly
multilingual).
