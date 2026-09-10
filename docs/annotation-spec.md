# Annotation specification

## Decision being annotated

Each query gets exactly one intent label:

| Intent | AI-eligible | Definition |
| --- | --- | --- |
| `question` | yes | A direct question: interrogative word or auxiliary at the start, a trailing `?`, or an indirect question phrased as a statement the user expects answered. |
| `exploratory` | yes | A request to explain, compare, assess, describe, or review a topic, including topic phrases such as "impact of X on Y", "causes of X", "trends in X". |
| `claim` | yes | A factual statement or assertion the user expects checked against the evidence (supports / contradicts / not addressed), e.g. "Air pollution causes premature deaths". |
| `retrieval` | no | Keyword or document retrieval: single terms, acronyms, short noun phrases, document-type words (report, PDF, dataset, directive, map...), year phrases, implicit data lookups. |
| `unknown` | no | Anything that does not clearly fit the classes above. Fails closed. |

Labeling is about **user intent**, not surface grammar: a statement without
auxiliaries can be a claim; a noun phrase like "causes of air pollution" is
exploratory, while "air pollution" is retrieval.

When two labels plausibly apply, use this precedence (matching the shipped
frontend policy): `question` > `exploratory` > `claim` > `retrieval` >
`unknown`.

## Languages

The canonical supported set (28) lives in
`src/eea_query_intent/languages.py`: the 24 official EU languages plus `is`
(Icelandic), `tr` (Turkish), and Norwegian as both `nb` and `nn`. A model or
dataset must be evaluated on every code in that list; generic `no` is not a
valid language value.

## Record schema

One JSON object per line:

| Field | Values | Notes |
| --- | --- | --- |
| `id` | unique string | `<lang>-<nnnn>` |
| `language` | supported code | |
| `text` | non-empty string | Exact query text, no markup |
| `intent` | one of the five labels | |
| `template_id` | string | Identifies the semantic template. Every translation or generation from the same template MUST share it. |
| `source_type` | `human_authored`, `synthetic_generated`, `synthetic_translated`, `real_anonymized` | |
| `review_status` | `unreviewed`, `policy_reviewed`, `native_reviewed` | |
| `split` | `train`, `validation`, `calibration`, `test` | |

## Hard dataset rules (enforced by the validator)

1. **Unique ids.**
2. **Template leakage protection:** one `template_id` may appear in exactly
   one split. Translations of the same semantic template must never straddle
   train/test (or validation/calibration/test), which is the main source of
   inflated per-language accuracy in translated corpora.
3. **Acceptance readiness:** a dataset used for final acceptance must have
   every record `native_reviewed` (`--require-acceptance-ready`).
4. **Local policy guards:** queries that fail the empty or word-count guard
   (default 20 words, Unicode-aware) are not classification targets and must
   not be part of the gold set.

## Review bar

- `policy_reviewed`: produced from an explicit policy (e.g. the English seed
  corpus) and checked against the definitions above. Acceptable for training.
- `native_reviewed`: reviewed by a fluent/native speaker of the query
  language against this spec, including the intent and the template grouping.
  Required for `validation`, `calibration`, and `test` records in any
  acceptance run.

## Suggested hard-negative inventory

Every language's test set should intentionally include: one-word queries,
acronyms and product/report names (SOER, PFAS, PM2.5), document-type + year
phrases, misspellings, missing diacritics, code-switching, URL-like tokens,
indirect questions, polite commands, claims that look like noun phrases, and
noun phrases that look like claims.
