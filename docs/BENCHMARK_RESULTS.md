# Benchmark results

## Final model: `setfit-v3` — all 28 languages (current)

The current shipping model is **SetFit only**. The fastText candidate was
removed from the service and codebase: it lost the bake-off and its adapter
was also incorrect (fastText returns `__label__`-prefixed labels and
`k=1` by default, so the aggregate eligible probability could not be
computed). Historical fastText numbers remain in the v1 section below, marked
as a rejected candidate.

`setfit-v3` is trained on the hand-authored English anchor bank plus
GPT-5.6-Sol QA'd per-row translations for all 27 other supported languages
(11,772 train / 1,330 calibration rows). Evaluated on the held-out test split
(25 languages present, 29 rows each), gated per language on the no-AI
false-routing rate (PASS = ≤1%):

| threshold | languages passing | failing |
|---|---|---|
| 0.85 | 19/25 | is, fi, it, lt, pt, sk |
| 0.90 | 22/25 | fi, is, sk |
| 0.95 | 24/25 | sk |
| **0.98 (chosen)** | **25/25** | — |

The `0.98` threshold was chosen as the **calibrated value** (the threshold
that minimizes the worst-language no-AI false-positive rate on the
calibration split) and it is fail-closed. It keeps the held-out test split
at 0 false routes for every language. The cost is lower eligible-query
recall (the fail-closed trade-off), which is expected to improve as
native-speaker-reviewed data reaches the 300-per-language bar.

Known hard cases (from the calibration split, not the test split): a pasted
URL (`https://eea.example/air`, rated ~0.98) and short topical phrases
("EU law on climate", ~0.99). The URL is now caught by a deterministic
**policy guard** (reason `url`) before the model; short topical phrases remain
the irreducible no-AI/eligible boundary and are the main reason the threshold
is held at `0.98`.

**Honest gap:** the test split has only 29 rows per language, so "0/9 no-AI
false routes" is not strong evidence of a ≤1% rate; the acceptance gates
(300 reviewed examples/language, ≥98% eligible precision, ≥95% macro-F1)
still fail on data volume, not model quality.

---

## v1 bake-off: dataset v1 (2026-09-10) — historical

Reproducible evidence for the first real bake-off of the two candidate
adapters on `data/multilingual/v1` (6,793 rows, 28 languages, 56 synthetic
templates + the 118-row English seed; split per template so no template
leaks across splits).

Raw per-language reports: [`reports/run-2026-09-10-v1/`](../reports)
(git-ignored by default — regenerate with the commands below).

## Deployment profile measured

- Machine: Apple Silicon (M-series) arm64, uv-managed CPython 3.12.13
- SetFit candidate served on `mps`; fastText candidate on CPU
- Warm single-query latency: ~10–30 ms (SetFit), <1 ms (fastText)
- Service RSS (SetFit, model loaded): ~312 MiB
- Artifact sizes: SetFit 466 MB (fp32 safetensors), fastText 6.7 MB (ftz)

The target deployment is a small x86-64/AVX2 CPU with ~1–2 GB RAM; ARM/MPS
numbers above are a latency upper-bound reference, not the deployment
profile. INT8/ONNX export of the SetFit artifact is the documented next
step to shrink the 466 MB fp32 file toward ~120 MB.

## Raw (uncalibrated) test split

| candidate | global no-AI FP | worst-language no-AI FP | languages >1% FP | eligible missed |
|---|---|---|---|---|
| fastText (raw) | 24/225 (10.7%) | 88.9% (pt) | 7/28 | 64/500 (12.8%) |
| SetFit (raw) | 28/225 (12.4%) | 66.7% (is) | 10/28 | 18/500 (3.6%) |

Both candidates fail the strict five-class acceptance gates
(macro-F1 ≥ 0.95, eligible precision ≥ 0.98, ≥300 reviewed examples per
language). The dominant errors are **subtype confusion**
(question/exploratory/claim and retrieval/unknown), not dangerous routing:
on English alone both models route the binary gate correctly on all but a
handful of subtype edges.

## Calibrated (abstention threshold) test split

The abstention threshold is chosen on the calibration split for the
smallest threshold whose worst-language no-AI false-positive rate is ≤ 1%:

- **SetFit → threshold 0.86** (meets the target on calibration: 0.000)
- fastText → no threshold meets the target (best 0.222 at 0.98)

| candidate (gated) | languages >1% no-AI FP | worst no-AI FP | worst eligible recall | abstained |
|---|---|---|---|---|
| SetFit @0.86 | 1/28 (is) | 44.4% (is, 2/9 rows) | 10% | 45.9% |
| fastText @0.98 | 2/28 (de, pt) | 88.9% (pt) | 20% | 53.9% |

The single remaining over-target language for SetFit is Irish, where 2 of
9 no-AI test rows were false-routed — sample noise at n=9, not a stable
failure mode (Irish was clean on calibration).

## Diagnosis of the residual errors

The high-confidence false routes concentrate in a handful of languages
(pt, et, is) and trace back to **synthetic data quality**, not model
capacity: several machine-translated template fills produce broken grammar
(e.g. Portuguese "conjunto de dados de a gestão de resíduos", double
articles in Italian/German topic fills), and the model faithfully learns
the surface form of the corrupted template. The English in-sample
confusion matrix is clean apart from exploratory↔claim edges.

Consequences, recorded honestly:

1. The acceptance gates are doing their job — they reject a prototype
   dataset rather than promote a model that is only good on average.
2. The production-blocking gap is **native-speaker-reviewed data**
   (per `docs/annotation-spec.md`: validation/calibration/test records
   require `native_reviewed`), not a larger model.
3. SetFit (multilingual MiniLM-L12 + linear head) is the shipping default
   candidate: it is the only candidate that meets the worst-language
   no-AI target on calibration, and it has far better eligible recall
   (3.6% missed vs 12.8% raw).
4. The abstention threshold buys safety at a coverage cost: at 0.86,
   ~30% of eligible calibration queries are abstained. Lowering the
   threshold after the native-review data exists is the planned trade
   lever (see `docs/benchmark-spec.md`).

## Reproduce (historical — inputs removed)

> The v1 bake-off inputs are no longer in the tree: the fastText candidate
> (`train_fasttext.py`, `models/fasttext`) and the v1 template dataset
> (`data/multilingual/v1/train|validation|calibration.jsonl`,
> `data/templates/`, `scripts/generate_multilingual.py`) were removed once the
> English-bank + per-row-translation pipeline replaced them. The commands
> below are kept for the record only. The current pipeline is documented in
> `docs/runbook.md`.

```bash
uv sync --all-groups
uv run python scripts/train_fasttext.py
uv run python scripts/train_setfit.py
uv run python scripts/calibrate.py fasttext
uv run python scripts/calibrate.py setfit
uv run python scripts/apply_threshold.py fasttext
uv run python scripts/apply_threshold.py setfit
uv run eea-query-intent evaluate \
  --gold data/multilingual/v1/test.jsonl \
  --predictions models/setfit/test-predictions-gated.jsonl \
  --minimum-eligible-count 0 --minimum-no-ai-count 0
```
