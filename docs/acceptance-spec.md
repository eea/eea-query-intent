# Acceptance corpus specification (v1)

The multilingual v1 test split (29 rows/language) is a regression set, not
statistical evidence: with ~9 no-AI rows per language, "0 false routes" is
compatible with a true false-routing rate up to ~33% (rule of three: 95% CI
upper bound ≈ 3/n). The acceptance gates in `metrics.py` therefore require
**at least 300 eligible and 300 no-AI examples per language** before a
language can pass. This corpus closes that gap.

## Design goals

1. **Held out**: no row (case-insensitive) may appear in any training,
   calibration, or regression file; every row has a unique `template_id`
   (`acc-<lang>-<intent>-<seq>`) that appears in no other split.
2. **Realistic**: query shapes are grounded in the actual EEA site
   (topics, datahub, indicators, SOER, AQI, WISE, ClimateADAPT, country and
   year patterns, document-type words) rather than template slot-filling.
3. **Native per language**: rows are generated *in* the target language by a
   strong LLM (GPT-5.6 Sol) acting as a native speaker, not translated from
   English — translation would re-import the 460-bank topic distribution and
   the machine-translation grammar breakage that v1 data suffered from.
4. **Labeled by user intent**, not surface grammar (see
   `annotation-spec.md`): a factual claim is `claim` (AI-eligible), a
   document/topic lookup is `retrieval` (no AI), a bare phrase is
   `retrieval`, out-of-scope/gibberish/code-switched fragments are `unknown`.
5. **Independently QA'd**: a separate GPT-5.6 Sol pass adversarially reviews
   every row (label correctness, native grammar, search-box register,
   duplicates) before rows are marked `llm_reviewed`.

## Composition (per language, 28 languages)

| intent       | target | generated (buffer) | notes |
| ------------ | ------ | ------------------ | ----- |
| question     | 150    | 165                | full natural-language questions |
| exploratory  | 120    | 135                | explain/compare/overview requests |
| claim        | 90     | 105                | checkable declarative statements |
| retrieval    | 350    | 365                | noun phrases, keywords, docs, maps, navigation |
| unknown      | 50     | 55                 | code-switched, gibberish, out-of-scope |
| **total**    | **760**| **825**            | buffer absorbs QA drops |

Totals: **760 rows/language × 28 = 21,280 rows** in
`data/acceptance/v1/test.jsonl` (per-language shards alongside).

The retrieval quota (350, ~46% of no-AI) is the class that the 1%
false-routing gate is about; unknown rows cover the long tail.

## Query-shape inventory (from EEA site research)

- **Topics**: air pollution (PM2.5, PM10, NO2, O3, SO2, ammonia), greenhouse
  gas emissions (CO2, LULUCF, ETS, Fit for 55, climate neutrality 2050),
  European Green Deal, water (Water Framework Directive, bathing waters,
  EQR, groundwater, floods, droughts), biodiversity (Natura 2000, protected
  areas, forests, deforestation, soil, pollinators), waste and circular
  economy (recycling, plastics, microplastics, e-waste, landfill), chemicals
  (PFAS, heavy metals, pesticides, POPs), noise, urban heat islands, ozone
  layer, renewable energy, transport/aviation emissions, agriculture
  (nitrogen), health effects.
- **Publications/products** (retrieval anchors): SOER, State of the
  Environment report, EEA indicators, datahub, emissions inventory, Air
  Quality Index, data maps, briefings, fact sheets, thematic reports,
  statistical databases, WISE, EIONET.
- **Modifiers**: EU/EEA country names, cities (Paris, Berlin, Warsaw,
  Milan, Athens, …), years (1990–2050), document words (PDF, report, map,
  data, statistics, database), navigation words (contact, about, careers).
- **Realism mix**: most retrieval rows are 1–6 words; a small minority of
  rows across all intents contain a realistic typo (~5%); question rows
  vary interrogative and length; no pattern dominates.

## Integrity rules (checked by `scripts/validate_acceptance.py`)

1. Every row: non-empty, ≤ 20 words (the local policy cap), no duplicate
   within the file (case-insensitive).
2. Zero overlap (case-insensitive) with `data/expanded_v2/`,
   `data/english_only/`, `data/seed/`, `data/multilingual/v1/`, and the
   460-row English bank.
3. Per-language character-set sanity (native alphabet + allowed Latin
   acronyms/digits/punctuation).
4. Per-language, per-intent counts exactly equal the target table.
5. `review_status` is `llm_reviewed` only after the QA pass; the acceptance
   gate (`--require-acceptance-ready`) then accepts the file.

## Evaluation

Run the SetFit model over the merged test split at the deployed abstain
threshold (`scripts/predict_acceptance.py`, default from the model manifest),
then:

```sh
uv run eea-query-intent evaluate \
  --gold data/acceptance/v1/test.jsonl \
  --predictions models/setfit/acceptance-predictions.jsonl
```

Defaults enforce the real gates: ≥300 eligible and ≥300 no-AI per language,
≤1% no-AI false-routing, ≥98% eligible precision, ≥95% macro-F1. A
language that fails is a *measured* failure — the fix is more
native-quality data or a threshold/abstention decision, never folding the
acceptance rows into training (that would break the holdout).
