# ADR 0003: Accept setfit-v4 as the production model; binary gate as the acceptance criterion; accept the low-resource gap (mt/ga/is)

Status: accepted (2026-09-14)

## Context

- setfit-v4 (multilingual MiniLM-L12-v2 backbone + 5-label SetFit head,
  62,224 training rows) meets the frozen 28-language exam on the binary
  routing objective (ADR 0002): worst-language keyword false-positive rate
  1.0% (target ≤ 1%), average eligible recall 85.1%, worst eligible
  precision 97.9% (mt).
- The formal five-class gate (per-language macro-F1 ≥ 0.95) fails in all 28
  languages (average 0.52): the head separates question/exploratory/claim
  weakly. Production consumes only the binary gate; per ADR 0002 the
  fine-grained intents never change downstream behavior.
- Three low-resource languages have low eligible recall on the exam:
  mt 12.8%, ga 33.1%, is 45.3% (all with eligible precision ≥ 97.9% and
  false-positive ≤ 0.3%). They carry only the 413-row v3 legacy corpus:
  the GPT generation phase was paused (quota needed elsewhere) and the
  in-house Gemma model measured weak on exactly these languages (probe
  flaw rates 0.72–1.00).
- A zero-cost pilot (2026-09-14) translated the 3,000-row English corpus
  into each language with local MT (NLLB-200 1.3B for ga/is, Helsinki-NLP
  opus for mt — NLLB produces broken Maltese), retrained identically, and
  re-ran the frozen exam at the same 0.98 threshold. Recall rose only to
  18.9% / 39.7% / 46.1%; the other 25 languages were unchanged or better.
  Conclusion: the bottleneck is the encoder's low-resource representations,
  not data volume or fluency. More translated data will not fix it.
- The other eight GPT-held languages (cs, el, et, hu, lt, lv, sl, sv)
  already score 90.0–96.4% eligible recall on their legacy rows alone.

## Decision

1. **setfit-v4 is the accepted production model to ship.** It already covers
   all 28 languages. Any further retraining (e.g. resuming the GPT phase for
   the other held languages) is a future improvement, not a release blocker.
2. **The production acceptance criterion is the binary routing gate**
   (ADR 0002): per-language keyword false-positive rate ≤ 1% and high
   binary eligible precision at the frozen 0.98 threshold. Five-class
   macro-F1 stays in reports as a diagnostic only, not a release gate.
3. **The mt/ga/is gap is accepted as product scope** (user decision, 2026-
   09-14): a large share of genuine questions in those three languages gets
   no AI summary. The failure mode is safe — search is unaffected and keyword
   searches never get a wrong summary. These three languages are excluded
   from any future GPT resume; no translation work is planned for them.
4. **The GPT data phase remains paused.** `.pipeline/gpt_paused` stays in
   place. The mt/ga/is checkpoints in `.pipeline/gpt_pending.txt` will not
   be resumed even if the pause is lifted. Resuming the other eight held
   languages (cs, el, et, hu, lt, lv, sl, sv — already 90.0–96.4% recall)
   remains an optional future improvement, not required for shipping.
5. If mt/ga/is quality ever becomes important, the structural lever is a
   backbone re-benchmark (ADR 0001 process) toward an encoder with stronger
   low-resource representations — not more data.

## Consequences

- The frontend integration (volto-searchlib `/_qi` proxy, ticket 307513)
  ships against setfit-v4 with no further model work.
- Deployment is purely infrastructure: the trained artifact
  (`models/setfit/`, ~450 MB, gitignored) must be packaged into or
  published to the rancher service image, and `QUERY_INTENT_SERVICE_URL`
  set on the Volto app.
- Acceptance reports keep showing `passes: false` against the five-class
  gate; this is expected, not a regression.
- The pilot scripts (`scripts/pilot_*`) are retained as the reproducible
  recipe for future translation experiments.
