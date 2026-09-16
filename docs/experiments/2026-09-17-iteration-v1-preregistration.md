# Iteration v1 — Pre-registration (2026-09-17)

This document locks the datasets, gates, selection rules, and transformation
log for the new classifier iteration BEFORE any training starts. It exists so
no later decision can quietly tune against the canonical exam.

Status: LOCKED for the data/build phases. Changes require a version bump here
and owner approval.

## 1. Scope

New production classifier for eea.europa.eu search (28 language codes; 26 site
languages). Binary routing: question/exploratory/claim are AI-eligible;
retrieval/unknown are not. Fail-closed: any abstention, error, or timeout
means no AI summary.

Supersedes: the D-1 promotion path of 2026-09-16/17 (same data recipe, but
the canonical exam and the 8 starved-language corpora change, so the old
waiver and old exam numbers are void).

## 2. Constraints (hard)

- No humans in the pipeline: no native speakers, no human labelers, no human
  review gates. Toolchain = assistant LLM (final reviewer, intermediate-
  advanced fluency, documented limitation) + in-house Gemma 31B (flagging
  layer only, never ground truth) + NLLB-1.3B / opus-mt-en-mt (translation)
  + deterministic scripts.
- No GPT API. No paid APIs.
- Deployment envelope: 2 vCPU / 2 GiB CPU-only, warm p95 <= 150 ms,
  frontend client timeout 2 s fail-closed.

## 3. Training data (strata — always reported separately)

| Stratum | Languages | Rows (expected) | Provenance |
|---|---|---|---|
| A: in-house native | bg da de en es fi fr hr it nb nl nn pl pt ro sk tr (17) | ~3,000 each | Gemma-generated, QA'd, committed (data/training/v1/) |
| B: GPT-native raw, QA'd this iteration | cs el et hu (3,000 each), lt lv (1,950 each) | 14,850 | GPT-5.6-luna generated pre-pause, on disk as *.raw.jsonl since 2026-09; QA via the hygiene protocol (section 7) instead of the GPT QA step (GPT paused) |
| C: machine-translated | sl sv | ~2,900 each | NLLB-1.3B translation of data/training/v1/en.jsonl (3,000 rows), source_id lineage, section 8 |
| D: accepted-weak (ADR 0003) | mt ga is | 413 legacy each + short-bank rows | No further data work for these three |
| E: short bank | all 28 | 160-170 each | assistant-authored English (170 concepts), NLLB/opus translated, committed recipe |
| Legacy 413 (8 langs) | cs el et hu lt lv sl sv | 413 each | v3-era translations, carried in expanded_v3; hygiene decides keep/drop per row (diversity test vs the new corpora) |

mt/ga/is are CLOSED for data work per ADR 0003 (gap accepted; encoder is the
bottleneck).

## 4. Calibration set

- Existing 1,330 rows (data/expanded_v3/calibration.jsonl) — hygiene-scanned.
- Extension: ~100 short (2-5 words) rows PER LANGUAGE (28 languages),
  weighted toward hard negatives (short keyword rows containing
  question/exploratory/claim trigger vocabulary).
- Exam and calibration concept pools are authored disjoint and split BEFORE
  translation. Lineage ids retained.
- Role: threshold selection + candidate/seed selection surface (the dev
  surface together with the probe batteries). Never the canonical exam.

## 5. Canonical exam ("v1 of the new iteration"; on disk: data/acceptance/v2/)

- KEEP all 21,280 existing rows (data/acceptance/v1/test.jsonl), subject to
  objective-defect and train-collision handling by hygiene.
- ADD per language x 28: 30 short questions, 30 short exploratory, 30 short
  claims (2-5 words), plus an adversarial short-negative slice (short
  retrieval/unknown rows containing trigger vocabulary). The positive slice
  is a DIAGNOSTIC slice: decisions use pooled/combined rates with reported
  uncertainty (one row = 3.3% of a class).
- The 1% gate denominator per language must hold >= ~300 short no-AI rows
  (existing short retrieval rows already number ~350/language; top up where
  thin).
- Source: assistant-authored fresh English concepts (disjoint concepts and
  templates from the 170-row training bank; common domain vocabulary
  allowed), NLLB/opus translated to 27 targets, strict dedup against every
  training row at surface AND concept level.
- Known limitation (accepted): the new short slice shares the NLLB engine
  with training stratum E; it is reported separately as an NLLB-derived
  short-query slice. The retained 21,280 GPT-era rows stay the independent
  majority.
- Freeze: version + sha256 of the manifest recorded here on freeze. Old
  comparability with pre-iteration exam results is DISCARDED by owner
  decision (2026-09-17). Archived models may be re-run on it for reference
  only.

## 6. Gates and threshold

- Gate: worst-language keyword false-positive rate <= 1% (retrieval+unknown
  rows routed AI-eligible).
- Threshold: re-derived, NOT reused. Procedure: sweep candidate thresholds
  on the calibration set (short-extended); pick the SMALLEST threshold
  satisfying the gate; verify on the canonical exam. 0.98 is a prior, not a
  constant.
- Candidate/seed ranking: among candidates meeting the FP constraint on the
  calibration surface, rank by eligible RECALL (not by lowest FP — that
  rewards unnecessary abstention). Probe batteries are reference.
- Waiver policy: NO inherited waivers. The 2026-09-16 Spanish 1.3% waiver
  is void for this iteration. A waiver may only be granted fresh by explicit
  owner decision, per model, per exam.

## 7. Data hygiene protocol (no humans)

Applies to: all training strata, calibration (existing + extension), exam
(existing + new rows) — before any of it is frozen/used.

1. Deterministic scan (scripts/hygiene_scan.py): NFC normalization diffs,
   invisible/control/bidi characters, garbled runs (U+FFFD, bullets, ratio
   > 5%), expected-script RATIO per language (not hard reject; unknown-class
   rows exempt from script flags — they intentionally contain off-language
   junk), length contract (>20 words / >500 chars), empties, exact
   same-language dupes, cross-language exact dupes >3 words, CROSS-LABEL
   CONFLICTS (same text, different intent — highest severity), intent
   sanity (non-question rows ending '?', question rows >8 words without '?'
   and no initial interrogative), exam collisions, MT source-identical
   checks (<=3 words: legitimate, keep+flag; >=4 words: drop).
2. Language-identifier sweep (fasttext lid.176) as a flag only — 2-5-word
   queries are below its reliable range.
3. In-house Gemma batch-judges flagged rows plus a stratified random sample
   (language x class x source), BLIND: it guesses the intent before seeing
   the stored label.
4. Assistant LLM reviews every flagged row; final keep/fix/drop. Uncertain
   rows are quarantined (dropped), not "repaired".
5. Report per language: scanned / flagged / dropped with reasons; decisions
   logged to data/hygiene/<date>/ for audit.

## 8. Transformation log (deterministic, recorded)

- Lowercasing: applied to ALL mix rows (text field) at mix build; service
  lowercases incoming queries. Originals retained in corpora.
- De-masking: ~75% of question rows ending in a question mark lose it;
  every 4th (1-based, per language x class) keeps it. Deterministic.
  Language-aware mark set: default "?" ; Greek also ";".
  Applied to strata A and B (and C where the translation carried the mark).
- Mix builder: scripts/v1_make_mix.py (new; the noq_* scripts remain as
  provenance of the previous iteration).

## 9. Training and selection

- Recipe: SetFit, backbone sentence-transformers/paraphrase-multilingual-
  MiniLM-L12-v2 (for the D-1-class baseline; backbone candidates come in a
  later phase, section 11), 5 labels, 2 epochs, batch 64, body LR 1e-5,
  head LR 1e-2, dropout 0 (MPS constraint).
- Seeds: 1, 2, 3 (fixed). Three runs per serious candidate.
- Selection: on the calibration surface (FP constraint first, then recall),
  never on the canonical exam.
- Canonical exam: run EXACTLY ONCE on the locked finalist (plus optional
  reference run of the incumbent production model). No tuning after.

## 10. Reproducibility record

- All corpora, concept pools, manifests, and scripts committed.
- MT decode settings: facebook/nllb-200-1.3B, float16 on mps, num_beams 5,
  max_new_tokens 64, src eng_Latn, forced_bos per target (sl slv_Latn,
  sv swe_Latn).
- sha256 of every frozen file recorded in the freeze note at the bottom of
  this document (added per freeze event).
- Before any HuggingFace release: verify NLLB/opus data-provenance and
  licensing statements for the model card.

## 11. Later phases (not part of this freeze)

- Binary routing head experiment (same data, binary objective).
- Backbone rebake-off per ADR 0001 (stronger multilingual encoders, full
  28-language mix, 3 strata reported, serving-stack benchmark
  latency/memory BEFORE canonical exam).
- Promotion, versioning, HF push, image build-time fetch (deferred).

---

## Freeze log

- 2026-09-17: document locked (pre-build).
  (manifest hashes appended as each artifact freezes)
