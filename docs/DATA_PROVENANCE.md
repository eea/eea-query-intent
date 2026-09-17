# Data and model provenance

This document is the content of the Hugging Face model card for
`query-intent-setfit-v1`. It must accompany the published model repo — the
training data contains machine-translated rows under non-commercial licenses,
so the provenance is a publication precondition.

## Model

- **Architecture**: SetFit linear head (384 -> 5 labels) on the
  `intfloat/multilingual-e5-small` encoder (BERT, 112M params, 250k vocab).
- **Labels**: `question`, `exploratory`, `claim`, `retrieval`, `unknown`.
- **Production routing**: binary — a query is AI-eligible when the sum of the
  three eligible-label probabilities is >= `abstain_threshold` (0.95).
  Anything below the threshold abstains and fails closed (no AI summary).
- **Training**: 2 epochs, batch 64, head LR 1e-2, backbone LR 1e-5, seed 3
  (selected from seeds 1-3 on the calibration set, ADR 0001 process).
- **Input normalization**: the service casefolds every query; all training
  text is casefolded.

## Training data (81,404 rows, `data/pilot/v1/train.jsonl`)

| stratum | rows | source | license note |
|---|---|---|---|
| A: in-house generated | ~50,700 | EEA in-house Gemma 31B, 17 languages, native generation | internal, AI-generated |
| B: GPT-native | ~16,800 | GPT-5.6-luna, 6 languages (cs el et hu lt lv), native generation | OpenAI ToS: outputs owned by user, training-data redistribution not permitted without review |
| C: machine-translated | ~5,800 | NLLB-200-1.3B (facebook) from the 3,000-row English corpus, sl + sv | **NLLB-200 is CC-BY-NC-4.0 (non-commercial)** |
| C2/D: legacy v3 | ~4,500 | English-anchor translations from the v3 corpus era, 11 GPT-pending languages | translated with NLLB/opus; opus-mt-en-mt (Maltese) is CC-BY-SA-4.0 |
| E: short bank | ~4,700 | hand-authored English, NLLB/opus-translated to 27 targets | see C |

Calibration: 4,008 rows (`data/pilot/v1/calibration.jsonl`), the 1,330-row
pre-v1 set plus 2,678 machine-translated hard-negative/short rows.
Exams: v1 (21,280 rows, GPT-generated, frozen) and v2 (23,718 rows = v1 plus
2,438 machine-translated short rows, frozen).

## Required disclosures for publication

1. **NLLB-200-1.3B is CC-BY-NC-4.0** — the model's training data contains
   NLLB translations (strata C, D, E and the exam-v2/calibration short rows).
   Non-commercial license: fine for the EEA (public body), must be declared
   on the model card; restricts commercial redistribution of the weights.
2. **opus-mt-en-mt (Maltese) is CC-BY-SA-4.0** — share-alike.
3. **GPT-generated rows** (stratum B) — OpenAI's terms govern; declare.
4. **Backbone** `intfloat/multilingual-e5-small` — verify its license
   (intfloat models are typically CC-BY-NC or Apache-2.0 per repo; check the
   HF repo's license field before publishing).
5. SetFit library is Apache-2.0; this repo is Apache-2.0.

## Exam results (frozen exam v2, threshold 0.95)

See `docs/benchmark-results/setfit-v1.json`: average eligible recall 0.859,
worst-language no-AI false positive 0.5% (Maltese, 2 of 400), average
abstention 54.6%.
