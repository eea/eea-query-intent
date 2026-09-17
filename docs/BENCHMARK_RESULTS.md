# Benchmark results

## Current: setfit-v1 (production)

SetFit linear head on `intfloat/multilingual-e5-small`, threshold **0.95**,
trained on the frozen 81,404-row mix (`data/pilot/v1/train.jsonl`).

Canonical measurement — frozen exam v2 (23,718 rows), single inference pass:

| metric | value |
|---|---|
| average eligible recall | **0.859** |
| worst-language no-AI false positive | **0.005** (Maltese, 2/400) — inside the 1% gate |
| average abstention | 0.546 |

Full per-language table: **`docs/benchmark-results/setfit-v1.json`**
(raw predictions in `models/setfit/exam-v2-predictions.jsonl`; any other
gated view re-derives deterministically via `scripts/sweep_acceptance.py regate`).

Threshold provenance: the pre-registered calibration-only derivation was in
fallback mode (the 1% worst-language gate is unreachable on the 4,008-row
calibration for every candidate family); 0.95 was chosen from the single
exam-v2 sweep — worst-language FP 0.005, average recall 0.859 — a
user-approved deviation recorded in `configs/setfit-v1.yaml`.

## Model history

| model | backbone | outcome |
|---|---|---|
| setfit-v3 | multilingual MiniLM-L12 | superseded; fastText lost the ADR 0001 bake-off (rejected candidate) |
| setfit-v4 | multilingual MiniLM-L12 | production until 2026-09-17; acceptance gate decided in ADR 0003; kept as rollback artifact |
| **setfit-v1** | multilingual-e5-small | **current production** (clean version name; iteration-v1 e5 finalist, seed 3) |

The ADR 0001 bake-off process, the ADR 0002 binary-gate design, and the
iteration-v1 pre-registration are in `docs/adr/` and
`docs/experiments/`. Historical fastText numbers from the v3 era are in git
history (removed from this document with the v4-era content).
