# ADR 0001: Benchmark-first model selection for the intent classifier

Status: accepted (2026-09-10)

## Context

The mandate is to replace the programmatic (rule-based) query
classification in `volto-searchlib` with an NLP solution that classifies
queries in all EEA languages (24 official EU languages + Icelandic,
Turkish, Norwegian) on a small CPU (1–2 vCPU, 1–2 GiB RAM).

An earlier proposal committed to fine-tuned `xlm-roberta-base` before
verification. Primary-source review showed that was not justified:

- XLM-R base is 279M parameters (~1.12 GB FP32), not ~116M, and its
  published 100-language list omits Maltese.
- `jhu-clsp/mmBERT-small` (140M total / 42M non-embedding, MIT) is the
  only genuinely small encoder whose published inventory explicitly
  includes all required languages, but its speed evidence is GPU-only and
  its low-resource behavior is unproven on this task.
- No public benchmark covers our four/five labels across all 28 languages
  on CPU.

## Decision

No production model is selected in advance. We build a bake-off harness in
this repository and promote the smallest candidate that meets the
worst-language acceptance gates (≤1% no-AI false-positive rate, ≥98%
eligible precision, ≥95% macro-F1, per-language minimum sample counts,
plus latency/RSS/cold-start gates) on the target hardware.

Candidates and roles are defined in `docs/benchmark-spec.md`. XLM-R serves
as the quality ceiling / offline teacher, not the default deployment.

## Consequences

- The service layer must stay model-agnostic (predict → `ClassificationResult`)
  until the bake-off finishes.
- Dataset and evaluation infrastructure is built before any training, which
  is the true critical path (per-language native review).
- If no compact model passes, we fall back to a fastText → compact
  transformer cascade with abstention, rather than relaxing the gates.
