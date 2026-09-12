"""GPT-5.6-Sol generation of the per-language acceptance corpus.

Generates, for one language, realistic EEA site search-box queries in the
target language (not translations), grouped by intent:

    question     165  (target 150)
    exploratory  135  (target 120)
    claim        105  (target 90)
    retrieval    365  (target 350)
    unknown       55  (target 50)

Rows are written to ``data/acceptance/v1/<lang>.raw.jsonl`` with
``review_status: unreviewed``; the QA pass (``gpt_acceptance_qa.py``) is what
promotes them to ``llm_reviewed`` and prunes to the target counts.

Usage: uv run python scripts/gpt_acceptance_gen.py <lang> [batch_size]
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Model tiers (Codex credit rates per 1M output tokens: Sol 750, Terra 375,
# Luna 150). All models draw from the same plan pool, so cheaper models
# stretch the rolling usage window: high-volume generation runs on Luna,
# quality-review passes run on Terra. Override via environment if needed.
GEN_MODEL = os.environ.get("EEA_QI_GEN_MODEL", "openai-codex/gpt-5.6-luna")
QA_MODEL = os.environ.get("EEA_QI_QA_MODEL", "openai-codex/gpt-5.6-terra")
MODEL = GEN_MODEL
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "acceptance" / "v1"

NAMES = {
    "bg": "Bulgarian",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "ga": "Irish",
    "hr": "Croatian",
    "hu": "Hungarian",
    "is": "Icelandic",
    "it": "Italian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mt": "Maltese",
    "nb": "Norwegian Bokm\u00e5l",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sv": "Swedish",
    "tr": "Turkish",
}

# (generate, target) per intent — the buffer absorbs QA drops.
QUOTAS = {
    "question": (165, 150),
    "exploratory": (135, 120),
    "claim": (105, 90),
    "retrieval": (365, 350),
    "unknown": (55, 50),
}

TOPICS = (
    "air pollution, PM2.5, PM10, nitrogen dioxide (NO2), ozone, sulphur dioxide, "
    "ammonia, greenhouse gas emissions, CO2, climate neutrality 2050, European "
    "Green Deal, Fit for 55, Emissions Trading System, LULUCF, renewable energy "
    "(solar, wind), Water Framework Directive, bathing waters, European Water "
    "Quality (EQR), groundwater, floods, droughts, biodiversity, Natura 2000, "
    "protected areas, forests, deforestation, soil, pollinators, waste, "
    "recycling, plastics, microplastics, e-waste, landfill, circular economy, "
    "PFAS, heavy metals, pesticides, persistent organic pollutants, noise, "
    "urban heat islands, ozone layer, transport emissions, aviation emissions, "
    "agriculture, nitrogen, health effects of pollution"
)

COUNTRIES = (
    "Romania, Germany, France, Poland, Italy, Spain, Greece, Hungary, Czechia, "
    "Sweden, Austria, Netherlands, Belgium, Portugal, Ireland, Finland, Denmark, "
    "Croatia, Slovenia, Slovakia, Bulgaria, Lithuania, Latvia, Estonia, Malta, "
    "Cyprus, Luxembourg"
)

CITIES = "Paris, Berlin, Warsaw, Milan, Athens, Madrid, Vienna, Prague, Bucharest, Rome, Lisbon, Budapest"
YEARS = "1990, 2000, 2005, 2010, 2015, 2019, 2020, 2022, 2024, 2025, 2030, 2050"

DEFINITIONS = {
    "question": (
        "A complete natural-language question that expects an explanation or "
        "answer: it starts with an interrogative word, uses inversion (Is / "
        "Does / Can / Will ...), or ends with '?'. It is NOT a document lookup. "
        "Examples (calibration only, in English): 'What is the main source of "
        "air pollution in Europe?'; 'How does climate change affect migratory "
        "birds?'; 'Which EU country emits the most CO2 per capita?'; 'Is "
        "plastic pollution getting worse in the Mediterranean?'; 'Why is the "
        "ozone layer recovering slowly?'; 'How much CO2 does the transport "
        "sector emit?'; 'When will the EU be climate neutral?'"
    ),
    "exploratory": (
        "A request for explanation, overview, comparison, trend or analysis \u2014 "
        "often an imperative (explain, compare, show, summarize, list, outline, "
        "tell me about) or 'what do we know about X'. It is NOT a closed yes/no "
        "question and NOT a bare noun phrase. Examples (calibration only, in "
        "English): 'Explain the EU Emissions Trading System'; 'Compare air "
        "quality in Berlin and Warsaw'; 'Overview of water quality in Europe'; "
        "'Show trends in EU renewable energy since 2010'; 'What do we know "
        "about PFAS in drinking water?'; 'Summarize the state of European "
        "forests'; 'List the main threats to European biodiversity'"
    ),
    "claim": (
        "A complete declarative sentence (subject + verb) asserting a checkable "
        "fact about the environment; the user wants it verified. No question "
        "marks, no imperatives, not a bare topic phrase. Examples (calibration "
        "only, in English): 'Air pollution causes premature deaths in "
        "Europe'; 'The EU has cut CO2 emissions by 30 percent since "
        "1990'; 'Electric cars make city air cleaner'; 'Nitrogen pollution "
        "harms biodiversity'; 'The Rhine is cleaner than it was 20 years "
        "ago'; 'Renewable energy can replace fossil fuels by 2030'"
    ),
    "retrieval": (
        "A document or topic lookup: a noun phrase, keyword string, "
        "publication name, or short navigation phrase. It must NOT be a full "
        "sentence: no interrogatives, no question marks, no complete "
        "declarative sentence. Typical lengths are 1-6 words. Shapes: a single "
        "topic word ('climate', 'biodiversity'); a topic pair ('water "
        "quality', 'emissions data'); topic + country ('air pollution "
        "romania'); topic + year ('ghg emissions 2024'); a bare acronym "
        "('PM2.5', 'SOER', 'EQR'); a publication name ('SOER 2025', 'state "
        "of the environment report'); a document/data phrase ('PDF on air "
        "quality', 'emissions inventory', 'statistical database', "
        "'bathing water data'); a map phrase ('map of air pollution in "
        "Europe', 'noise map'); a navigation word ('datahub', 'contact', "
        "'about the agency')"
    ),
    "unknown": (
        "Fragments that must NOT trigger an AI answer, in a realistic mix: "
        "about 40% code-switched fragments mixing the target language with "
        "English environmental terms (e.g. a half-translated query); about "
        "40% genuine out-of-scope queries about non-environmental topics "
        "(sports, cooking recipes, finance, celebrity news, a local weather "
        "forecast, movies); about 20% gibberish or corrupted strings. These "
        "are what real users accidentally type"
    ),
}

PROMPT_TMPL = """You are a native @NAME@ speaker. Generate exactly @N@ realistic search-box queries in @LANG_NAME@ for the European Environment Agency website (eea.europa.eu): the exact strings a real visitor would type into the site search.

Task type: @INTENT@
Definition: @DEFINITION@

Topic pool (adapt into @LANG_NAME@ with native conventions; acronyms and proper nouns such as PM2.5, PM10, NO2, CO2, SOER, EU, EEA, WFD, ETS, EQR, LULUCF, PFAS, Paris Agreement, Green Deal, city and river names stay as written):
@TOPICS@
Country pool: @COUNTRIES@
City pool: @CITIES@
Year pool: @YEARS@

Diversity rules:
- Vary the topic per row; no topic may appear more than 3 times in this batch.
- Do not repeat a phrasing pattern: no more than ~20% of rows may start with the same word or structure.
- Keep the search-box register: terse, no polite filler, no full essays.
- About 5% of rows may contain a single realistic typo.
- Every row must be at most 20 words and must be written in @LANG_NAME@ (acronyms, proper nouns, digits may stay Latin).
- No row may be identical to another row.

Output STRICT JSON only, no prose, no markdown: {"rows":["...","..."]} containing exactly @N@ strings.
"""

LANG_NOTES = {
    "nn": " Use Norwegian Nynorsk, NOT Bokm\u00e5l.",
    "is": " Use correct Icelandic diacritics (\u00f0, \u00fe, \u00e6, \u00f6, \u00ed, \u00fa, \u00fd, \u00e9, \u00e1).",
    "ga": " Use natural modern Irish.",
    "mt": " Use natural Maltese.",
    "el": " Use Greek with correct diacritics.",
    "tr": " Use Turkish; Turkish search queries are usually lowercase without capitalization of the first word.",
    "de": " German search queries are usually lowercase (no initial capital unless a proper noun).",
    "fr": " French search queries are usually lowercase (no initial capital).",
    "es": " Spanish search queries are usually lowercase (no initial capital).",
    "pt": " Portuguese search queries are usually lowercase (no initial capital).",
    "it": " Italian search queries are usually lowercase (no initial capital).",
    "nl": " Dutch search queries are usually lowercase (no initial capital).",
}


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e < s:
        raise ValueError("no JSON object in model output")
    return json.loads(text[s : e + 1])


def _pi_once(prompt: str, model: str = MODEL) -> str:
    # Pin the PATH: pi (shebang: #!/usr/bin/env node) must run on a known-good
    # node. Homebrew's node (v25.9.0_2) is broken on this machine (dyld: libllhttp
    # 9.3 missing after an llhttp upgrade), and launchd environments have no
    # node at all, so prepend the fnm-installed node and the bun bin dir.
    preferred_path = (
        "/Users/razvan/.local/share/fnm/node-versions/v22.21.1/installation/bin"
        ":/Users/razvan/.bun/bin"
    )
    env = dict(os.environ)
    env["PATH"] = preferred_path + ":" + env.get("PATH", "/usr/bin:/bin")
    out = subprocess.run(
        [
            "pi",
            "--model",
            model,
            "-p",
            "-nt",
            "-nc",
            "-ns",
            "-ne",
            "--no-session",
            prompt,
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    return (out.stdout or "") + "\n" + (out.stderr or "")


def call_gpt_raw(
    prompt: str, tries: int = 3, limit_waits: int = 60, model: str = GEN_MODEL
) -> dict:
    last_err = None
    for attempt in range(1, tries + 1):
        try:
            text = _pi_once(prompt, model)
            try:
                return extract_json(text)
            except Exception:
                # Non-JSON output: a usage/rate limit or a transient failure
                # (e.g. contention between parallel workers). Both are handled
                # the same way: long 5-minute backoff, many waits. Log the
                # raw reply once so the failure is diagnosable.
                print(
                    f"  non-JSON reply ({len(text)} chars): {text[:300].strip()!r}",
                    flush=True,
                )
                for wait in range(1, limit_waits + 1):
                    print(
                        f"  GPT call failed (limit/transient), waiting 5 min "
                        f"({wait}/{limit_waits})",
                        flush=True,
                    )
                    time.sleep(300)
                    text = _pi_once(prompt, model)
                    try:
                        return extract_json(text)
                    except Exception:
                        continue
                raise RuntimeError("GPT call failed after all waits") from last_err
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < tries:
                time.sleep(300)
    raise RuntimeError(f"GPT call failed after {tries} tries: {last_err}")


def call_gpt(prompt: str, tries: int = 3, model: str = GEN_MODEL) -> list[str]:
    last_err = None
    for _ in range(tries):
        try:
            res = call_gpt_raw(prompt, model=model)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(300)
            continue
        rows = [
            r.strip() for r in res.get("rows", []) if isinstance(r, str) and r.strip()
        ]
        if rows:
            return rows
        last_err = ValueError("empty rows in model output")
    raise RuntimeError(f"GPT gen failed after {tries} tries: {last_err}")


def build_prompt(lang: str, intent: str, n: int) -> str:
    prompt = (
        PROMPT_TMPL.replace("@N@", str(n))
        .replace("@INTENT@", intent)
        .replace("@DEFINITION@", DEFINITIONS[intent])
        .replace("@TOPICS@", TOPICS)
        .replace("@COUNTRIES@", COUNTRIES)
        .replace("@CITIES@", CITIES)
        .replace("@YEARS@", YEARS)
        .replace("@LANG_NAME@", NAMES[lang])
        .replace("@NAME@", NAMES[lang])
    )
    prompt += LANG_NOTES.get(lang, "")
    return prompt


def generate_intent(
    lang: str, intent: str, n: int, batch: int, avoid: set[str] | None = None
) -> list[str]:
    rows: list[str] = []
    seen = set(avoid or set())
    while len(rows) < n:
        need = min(batch, n - len(rows))
        got = call_gpt(build_prompt(lang, intent, need))
        # keep only the first `need` fresh rows (drop batch + earlier dups)
        fresh = []
        for r in got:
            key = r.casefold()
            if key not in seen:
                fresh.append(r)
                seen.add(key)
                if len(fresh) == need:
                    break
        rows.extend(fresh)
        print(f"{lang}/{intent} {len(rows)}/{n}", flush=True)
    return rows[:n]


def main() -> None:
    lang = sys.argv[1]
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 75
    if lang not in NAMES:
        sys.exit(f"unknown language {lang}; expected one of {sorted(NAMES)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{lang}.raw.jsonl"

    # crash-resilient resume: completed intents are skipped, the raw file is
    # rewritten after every intent so a mid-run crash loses at most one intent
    records: list[dict] = []
    if out.exists():
        records = [
            json.loads(line)
            for line in out.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    per_intent = {intent: 0 for intent in QUOTAS}
    for rec in records:
        per_intent[rec["intent"]] += 1
    avoid = {rec["text"].casefold() for rec in records}

    for intent, (n_gen, _target) in QUOTAS.items():
        have = per_intent[intent]
        if have >= n_gen:
            print(
                f"{lang}/{intent} already complete ({have}/{n_gen}), skipping",
                flush=True,
            )
            continue
        need = n_gen - have
        texts = generate_intent(lang, intent, need, batch, avoid)
        for seq, text in enumerate(texts, start=have + 1):
            row_id = f"acc-{lang}-{intent}-{seq:04d}"
            records.append(
                {
                    "id": row_id,
                    "template_id": row_id,
                    "language": lang,
                    "text": text,
                    "intent": intent,
                    "source_type": "synthetic_generated",
                    "review_status": "unreviewed",
                    "split": "test",
                }
            )
            avoid.add(text.casefold())
        with out.open("w", encoding="utf-8") as handle:
            for rec in records:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"{lang}/{intent} done ({per_intent[intent] + need}/{n_gen}), file saved",
            flush=True,
        )
    print(f"{lang}: wrote {out} ({len(records)} rows)", flush=True)


if __name__ == "__main__":
    main()
