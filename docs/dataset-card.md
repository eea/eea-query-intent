---
license: cc-by-nc-4.0
task_categories:
- text-classification
tags:
- query-intent
- multilingual
- eea
- synthetic-data
language:
- bg
- cs
- da
- de
- el
- en
- es
- et
- fi
- fr
- ga
- hr
- hu
- is
- it
- lt
- lv
- mt
- nl
- no
- pl
- pt
- ro
- sk
- sl
- sv
- tr
size_categories:
- 10K<n<100K
---

# EEA Query Intent — setfit-v1 training data

The **exact training mix** used to train
[`eeahugs/query-intent-setfit-v1`](https://huggingface.co/eeahugs/query-intent-setfit-v1),
the multilingual query-intent classifier for the [European Environment
Agency](https://www.eea.europa.eu) website search. 81,404 rows, 28 languages,
5 intent classes.

**License: CC-BY-NC-4.0 (non-commercial use only)** — see "Provenance and
licensing" below.

## Format

JSON Lines. One object per line:

| field | meaning |
|---|---|
| `id` | row id, `<prefix>-<lang>-<intent>-NNNN` |
| `intent` | one of `question`, `exploratory`, `claim`, `retrieval`, `unknown` |
| `language` | BCP-47-ish code, one of 28 |
| `text` | the query text, casefolded (lowercase) |
| `source_id` | lineage id of the source row (null for authored rows) |
| `source_type` | `synthetic_generated`, `synthetic_translated`, etc. |

Class distribution: question 24,598 · retrieval 21,102 · exploratory 15,964 ·
claim 15,946 · unknown 3,794.

## Training recipe (for reproducibility)

`SetFitModel` on `intfloat/multilingual-e5-small` (backbone revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3`), 2 epochs, batch 64, head LR 1e-2,
backbone LR 1e-5, fixed seed 3 (selected from seeds 1-3 on a held-out
calibration set). Five labels as above; the production route is binary —
AI-eligible when P(question)+P(exploratory)+P(claim) >= 0.95.

This file is the complete training input. The held-out calibration set and
the frozen acceptance exams are intentionally **not** published.

## Integrity

SHA-256 of this file (first 16 hex chars): `1e0e75eb65749835`

## Provenance and licensing

| stratum | rows (approx) | source |
|---|---|---|
| A | ~50,700 | in-house Gemma 31B native generation, 17 languages (AI-generated) |
| B | ~16,800 | GPT-5.6-luna native generation, 6 languages: cs, el, et, hu, lt, lv (declared: OpenAI model outputs, published with the owner's decision to include) |
| C | ~5,800 | **NLLB-200-1.3B machine translation** (CC-BY-NC-4.0), sl + sv |
| D | ~4,500 | legacy English-anchor translations, 11 languages (NLLB / opus-mt; the Maltese model is CC-BY-SA-4.0) |
| E | ~4,700 | hand-authored English short-query bank, NLLB/opus-translated |

Because the dataset contains machine-translation outputs from
**NLLB-200 (CC-BY-NC-4.0)** and **opus-mt-en-mt (CC-BY-SA-4.0)**, this dataset
is published under **CC-BY-NC-4.0**: attribution required, non-commercial use
only, share-alike for the opus-derived Maltese rows. The companion model
weights are published under MIT by the owner's decision; the upstream
disclosures in the model card apply to any reuse of this data.
