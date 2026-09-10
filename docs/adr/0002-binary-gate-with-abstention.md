# ADR 0002: Binary routing gate with abstention; no English-regex fallback

Status: accepted (2026-09-10)

## Context

The frontend only consumes `shouldGenerateAI` from the current classifier;
the fine-grained intents (`question`/`exploratory`/`claim`) do not change
downstream behavior. The cost of a false positive (an LLM call for a plain
keyword search) is much higher than the cost of a false negative (a
question that simply gets search results).

## Decision

1. The production objective is the **asymmetric binary decision**:
   AI-eligible (question, exploratory, claim) vs no-AI (retrieval,
   unknown), with an explicit **abstain** outcome when calibrated
   confidence is below threshold. Abstain routes to no-AI.
2. Fine-grained intents are still predicted, for annotation, monitoring,
   and per-intent recall reporting — but they never relax the binary gate.
3. On service error, timeout, or malformed response the frontend **fails
   closed** (no AI summary; search remains fully functional). We do **not**
   fall back to the English regex classifier, because that would make
   behavior language-dependent and silently keep the implementation we are
   replacing alive.
4. Empty-input and overlong-input checks stay as deterministic local
   policy guards (not linguistic classification) and run before any
   network call.

## Consequences

- The API returns `eligible`, `intent`, `confidence`,
  `eligible_probability`, `abstained`, and `model_version` so both the
  binary gate and diagnostics are available.
- Evaluation must track false-positive rate, eligible precision, and
  abstention/coverage per language (implemented in `metrics.py`).
- Thresholds are calibrated on the `calibration` split via the
  risk–coverage curve and frozen before the `test` split is used.
