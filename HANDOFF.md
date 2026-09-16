# HANDOFF — EEA query intent, iteration v1 overnight run

Written: 2026-09-17, ~01:45. Owner: user (Razvan). Previous handoff context:
the compacted session summary in the assistant's memory; this document is
self-contained for a fresh conversation.

## Incident 2026-09-17 ~02:10 (recovered, no data loss)

The first chain run (01:49) died seed-by-seed at the threshold step: the
v1 calibration/exam rows were missing the `template_id`/`source_type`/
`review_status`/`split` fields the package's dataset contract requires
(the builder emitted 5-key rows). Consequence: d1-s1 and binary-s1 were
fully trained (models + calibration predictions on disk) but no thresholds
were derived, and backbone never trained. Fixed:
- `v1_make_mix.py` now emits full 8-key contract rows (pool rows mapped to
  allowed source_type values, `llm_reviewed`, correct split) — data content
  unchanged, exam sha changed (e92055c1ff647940 -> 79fca1bb9edab755),
  exam was never run so no comparability impact; mix sha unchanged
  (1e0e75eb65749835, identical to the committed dataset).
- `choose_final_threshold.py` measure() now uses `uv run eea-query-intent`
  and validates the report instead of the exit code (evaluate exits 1 when
  the diagnostic five-class gate is unmet, which is expected).
- `v1_train_eval.sh` skips a seed whose model + calibration predictions
  already exist, so the restarted chain (02:25) reuses d1-s1 and binary-s1.
- Verified: both files pass `load_dataset` full-contract validation and
  threshold derivation runs end-to-end on d1-s1.

## Where things stand

- **Classifier production model is still setfit-v4** in `models/setfit/`,
  served by the launchd job `com.razvan.eeaki-service` on 127.0.0.1:8100.
  Verified healthy at 00:00 (uptime ~2.4 days, threshold 0.98). It is
  untouched by everything below and remains the rollback artifact.
- **Iteration v1 datasets are built and frozen** (pre-registration:
  `docs/experiments/2026-09-17-iteration-v1-preregistration.md`):
  - Training mix: `data/pilot/v1/train.jsonl` — 81,404 rows, fully
    lowercased (casefold), de-marked (~75% of marked questions), 45
    cross-label conflicts resolved by surface form, 21 adjudicated drops.
    Strata: A 17 in-house native, B 6 GPT-native raw (cs/el/et/hu 3000,
    lt/lv 1950), C MT sl/sv (2907/2889, NLLB-1.3B, `source_id` lineage),
    C2+D legacy v3 rows for the 11 GPT-routed languages (retention
    filtered), E short bank (~165-170/language). Manifest:
    `data/pilot/v1/manifest.json` (sha256 recorded).
  - Calibration: `data/pilot/v1/calibration.jsonl` — 4,008 rows (1,330 old
    + 2,678 new short pool), lowercased. This is the ONLY set the
    threshold is derived from.
  - Canonical exam: `data/acceptance/v2/test.jsonl` — 23,718 rows
    (21,280 frozen v1 rows untouched + 2,438 new 2-5-word short rows,
    original case, 28 languages, 4 concept quarantines applied).
    sha256 in `data/acceptance/v2/manifest.json`. **Runs exactly once,
    only on the locked finalist, never for tuning.**
- **Data hygiene is complete**: deterministic scan + fasttext-style
  flags + in-house Gemma blind judge (1,564 rows, 297 disagreements)
  adjudicated by the assistant. All decisions recorded in
  `data/hygiene/2026-09-17/drop_decisions.jsonl` (26 row-level + 4
  concept quarantines). No GPT was used; `.pipeline/gpt_paused` is still
  in place and GPT data generation remains paused.

## What is running right now (background, caffeinate-guarded)

1. `scripts/v1_family_chain.sh` (launched ~01:45) trains the three
   candidate families sequentially on MPS, seeds 1/2/3 each:
   - `d1` → `models/setfit-v1-s{1,2,3}` (five-class, the D-1-class mix)
   - `binary` → `models/setfit-v1b-s{1,2,3}` (two-label eligible head)
   - `backbone` → `models/setfit-v1e5-s{1,2,3}` (intfloat/multilingual-e5-small)
   Per family it derives each seed's threshold from calibration only
   (`choose_final_threshold.py`) and runs `v1_select_seed.py` (qualify on
   the worst-language 1% no-AI FP gate, rank by calibration recall).
   Log: `/tmp/v1_family_chain.log` + `/tmp/v1_train_eval_<family>.log`.
   Flags: `.pipeline/v1_<family>_train_done.flag`,
   `.pipeline/v1_all_families_done.flag` when all three are done.
   Expected total: ~2.5-3.5 h.
2. `scripts/v1_post_chain.sh` (launched right after) waits for the chain
   flag, locks the cross-family finalist (`v1_lock_finalist.py`, same
   calibration rule, writes `models/finalist/lock.json`), runs
   `v1_canonical_exam.py <finalist> <threshold>` **exactly once**
   (refuses if predictions already exist), writes the formal per-language
   report, copies everything to `reports/v1_overnight/`, and touches
   `.pipeline/v1_overnight_done.flag` (or `v1_overnight_failed.flag`).
   Log: `/tmp/v1_post_chain.log`. **It deliberately stops there.**

## What to do in the morning (in order)

1. Check `.pipeline/v1_overnight_done.flag`. Read
   `reports/v1_overnight/` (lock.json + canonical exam report) and
   `/tmp/v1_train_eval_*.log` for per-seed numbers.
2. Promotion decision package: compare the finalist's exam numbers
   against (a) setfit-v4 production baseline (exam v1 report in
   `reports/final_acceptance_report.json` — not directly comparable,
   different exam), and (b) the pre-registered gate (worst-language
   no-AI FP <= 1% at the re-derived threshold; NO inherited waivers —
   the 2026-09-16 one-language waiver is void).
3. Optional per the overnight autonomy grant: one-off `pi --model
   openai-codex/gpt-5.6-sol --thinking high` review of the promotion
   decision (review only; do NOT lift `.pipeline/gpt_paused`).
4. If promoting:
   - Version the model with a clean name (e.g. `setfit-v1`) for the
     future Hugging Face push (user's stated deployment plan: push the
     model to HF, image fetches it at build time).
   - Swap `models/setfit` -> the finalist (keep setfit-v4 as rollback
     copy, e.g. `models/setfit-v4-rollback/`).
   - **Atomic pair**: add the one-line query lowercasing in
     `src/eea_query_intent/service.py` `classify_query` (lowercase the
     input right before model.predict_proba) in the SAME step as the
     model swap — the lowercased model must never run without the
     service-side lowercasing and vice versa.
   - Restart the launchd job with the established absolute-path pattern
     (`launchctl remove com.razvan.eeaki-service`, then submit
     `/Users/razvan/Work/eea-query-intent/scripts/run_service.sh`),
     verify `/health` (model_version must be the new name) and run
     `uv run python scripts/noq_probe.py` (probe battery).
   - `uv run pytest` + `uv run ruff check` + commit the model swap and
     the service line together.
5. Deferred (do NOT start without the user): Docker image (blocked by
   the full disk, draft Dockerfile uncommitted), Hugging Face release
   (needs NLLB/opus licensing + provenance check first), backbone
   bake-off beyond the e5-small family already in the chain, exam-v2
   beyond the short slice, the 8 remaining GPT languages' GPT phase
   (paused).

## Hard invariants (do not break)

- Never consume GPT quota: `.pipeline/gpt_paused` stays; any GPT use is
  a one-off `pi` review call only, and only when a decision needs it.
- Fail-closed classifier: any timeout/error/abstention -> no AI summary.
- The frozen exam runs once on the locked finalist; never select or
  tune on it; the threshold comes from calibration only.
- No humans in the pipeline (no native speakers, no human reviewers);
  Gemma is a flagging layer, never ground truth.
- Do not touch the user's Volto dev-server processes.
- macOS/bash 3.2, rtk mangles multi-line/piped/looped bash-tool
  commands (write Python or script files instead), launchd jobs need
  absolute paths and vanish after a full shutdown, caffeinate guards
  for long runs (a shutdown once killed an in-progress translation).

## Repo state (branch `main`, no remote)

Committed up to the handoff commit (see `git log`). Untracked/deferred:
draft `Dockerfile` + `.dockerignore` (docker work paused), the old
candidate model dirs (`models/setfit-boundary*`, `setfit-en2`,
`setfit-noq-short` — cleanup candidates, ask before deleting),
`reports/` is gitignored. The volto-searchlib repo (separate git repo,
branch `feat/query-intent-api`) is clean and waiting for the user to
push it themselves; the deployed service URL env var is
`QUERY_INTENT_SERVICE_URL` (origin only).
