"""GPT-5.6-Sol translation QA pass.

For a language, load its TRANSLATIONS list (the source of truth), batch it,
ask the GPT-5.6 Sol model (via the pi CLI) to review each English->lang pair
as a native reviewer, apply the "fix" corrections, and rewrite the .py.

Usage: uv run python scripts/gpt_translate_qa.py <lang> [batch_size]
"""

import importlib.util
import json
import re
import subprocess
import sys

MODEL = "openai-codex/gpt-5.6-sol"

NAMES = {
    "et": "Estonian", "fi": "Finnish", "is": "Icelandic", "it": "Italian",
    "lt": "Lithuanian", "lv": "Latvian", "nl": "Dutch", "pt": "Portuguese",
    "sv": "Swedish", "tr": "Turkish",
}

SECTIONS = [
    (0, 120, "question"),
    (120, 220, "exploratory"),
    (220, 320, "claim"),
    (320, 420, "retrieval"),
    (420, 460, "unknown"),
]

PROMPT_TMPL = """You are a native @NAME@ (@LANG@) linguist doing QA on machine-drafted translations of environmental search queries.
For each numbered pair, judge the @LANG@ translation against the English source.
Rules:
- The English source is a SHORT SEARCH QUERY (not a full sentence for retrieval/exploratory rows). Keep the same short query register.
- Preserve the sentence type: questions stay questions, claims stay declarative statements, retrieval/exploratory stay noun phrases.
- Correct grammar, word order, case, agreement, gender, diacritics, and word choice. Make it how a native speaker would type this query.
- Do NOT "translate" rows that are intentional gibberish or code-switched fragments: bare nonsense words (e.g. zorpflimble, blargh), single foreign-language fragments from other languages, bare URLs, digit-only strings, or punctuation-only strings. Mark those ok and leave them unchanged.
- Keep proper nouns (PFAS, PM2.5, PM10, CO2, NO2, SOER, EU/EEA acronyms, city and river names) as in the source unless a native convention clearly applies.
Output STRICT JSON only, no prose: {"rows":[{"i":<num>,"status":"ok"|"fix","corrected":"<corrected @LANG@ only when fix, else empty string>"}]}
Pairs:
"""


def load_lang(lang):
    path = f"scripts/translations/{lang}.py"
    spec = importlib.util.spec_from_file_location(f"tr_{lang}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return list(m.TRANSLATIONS)


def english_for(lang):
    rows = [json.loads(l) for l in open(f"data/translations/{lang}.jsonl", encoding="utf-8")]
    return [r["english"] for r in rows]


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e < s:
        raise ValueError("no JSON object in model output")
    return json.loads(text[s:e + 1])


def call_gpt(pairs, lang, tries=3):
    name = NAMES[lang]
    body = PROMPT_TMPL.replace("@NAME@", name).replace("@LANG@", lang)
    for i, (en, tr) in enumerate(pairs, 1):
        body += f"{i} EN: {en} | {lang}: {tr}\n"
    last_err = None
    for attempt in range(tries):
        try:
            out = subprocess.run(
                ["pi", "--model", MODEL, "-p", "-nt", "-nc", "-ns", "-ne", "--no-session", body],
                capture_output=True, text=True, timeout=600,
            )
            return extract_json(out.stdout)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"GPT QA failed after {tries} tries: {last_err}")


def rewrite_py(lang, trs):
    name = NAMES[lang]
    lines = [
        f'"""{name} ({lang}) translations of the 460-row English anchor bank, in order."""',
        "",
        "TRANSLATIONS = [",
    ]
    for s, e, label in SECTIONS:
        lines.append(f"    # {s + 1}-{e} {label}")
        for i in range(s, min(e, len(trs))):
            lines.append(f"    {json.dumps(trs[i], ensure_ascii=False)},")
    lines.append("]")
    with open(f"scripts/translations/{lang}.py", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    lang = sys.argv[1]
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    trs = load_lang(lang)
    if len(trs) != 460:
        sys.exit(f"{lang}: expected 460 rows, got {len(trs)}")
    english = english_for(lang)
    if len(english) != 460:
        sys.exit(f"{lang}: expected 460 english rows, got {len(english)}")

    log = []
    for start in range(0, 460, batch):
        chunk = trs[start:start + batch]
        enchunk = english[start:start + batch]
        res = call_gpt(list(zip(enchunk, chunk)), lang)
        rows = {r["i"]: r for r in res.get("rows", [])}
        changed_in_batch = 0
        for i in range(len(chunk)):
            r = rows.get(i + 1)
            if not r:
                continue
            if r.get("status") == "fix" and r.get("corrected", "").strip():
                new = r["corrected"].strip()
                old = trs[start + i]
                if new != old:
                    trs[start + i] = new
                    log.append((start + i + 1, old, new))
                    changed_in_batch += 1
        print(f"{lang} batch {start + 1}-{start + len(chunk)}: {changed_in_batch} fixes", flush=True)

    rewrite_py(lang, trs)
    with open(f"reports/gpt_qa_{lang}.json", "w", encoding="utf-8") as f:
        json.dump([{"row": r, "old": o, "new": n} for r, o, n in log], f, ensure_ascii=False, indent=1)
    print(f"{lang}: {len(log)} total corrections applied", flush=True)


if __name__ == "__main__":
    main()
