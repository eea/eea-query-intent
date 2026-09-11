"""GPT-5.6-Sol direct translation generation.

Translate the 460-row English anchor bank into a target language and write
``scripts/translations/<lang>.py``. The nine intentional gibberish /
code-switched / URL / digit / punctuation ``unknown`` rows are forced
verbatim (copied from the English source) and never sent to the model.

Usage: uv run python scripts/gpt_translate_gen.py <lang> [batch_size]
"""

import json
import re
import subprocess
import sys
from pathlib import Path

MODEL = "openai-codex/gpt-5.6-sol"
ROOT = Path(__file__).resolve().parent.parent

NAMES = {
    "bg": "Bulgarian",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "es": "Spanish",
    "fr": "French",
    "ga": "Irish",
    "hr": "Croatian",
    "hu": "Hungarian",
    "mt": "Maltese",
    "nb": "Norwegian Bokm\u00e5l",
    "nn": "Norwegian Nynorsk",
    "pl": "Polish",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
}

SECTIONS = [
    (0, 120, "question"),
    (120, 220, "exploratory"),
    (220, 320, "claim"),
    (320, 420, "retrieval"),
    (420, 460, "unknown"),
]

# 0-based bank rows that are intentional gibberish / code-switched / URL /
# digits / punctuation and must stay verbatim in every language.
VERBATIM = {430, 431, 432, 433, 434, 435, 436, 438, 439}

PROMPT_TMPL = """You are a native @NAME@ (@LANG@) linguist. Translate each numbered English search query into @LANG@.
These are SHORT SEARCH QUERIES for an environmental website (not full sentences for retrieval/exploratory rows). Rules:
- Keep the short query register (a native speaker typing this into a search box).
- Preserve the sentence type: questions stay questions, claims stay declarative statements, retrieval/exploratory rows stay noun phrases.
- Use correct grammar, word order, case, agreement, gender, and diacritics.
- Keep proper nouns and acronyms (PFAS, PM2.5, PM10, CO2, NO2, SOER, EU, EEA, Green Deal, Paris Agreement, city/river names) as in the source unless a clear native convention applies.
- Do NOT translate or "fix" any row that is gibberish, a bare URL, digits, or punctuation; leave it exactly as given.
Output STRICT JSON only, no prose: {"rows":[{"i":<num>,"text":"<@LANG@ translation>"}]}
Rows:
"""


def load_bank() -> list[dict]:
    path = ROOT / "data" / "english" / "expanded_v1.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e < s:
        raise ValueError("no JSON object in model output")
    return json.loads(text[s : e + 1])


def call_gpt(rows: list[tuple[int, str]], lang: str, tries: int = 3) -> dict:
    name = NAMES[lang]
    body = PROMPT_TMPL.replace("@NAME@", name).replace("@LANG@", lang)
    for idx, en in rows:
        body += f"{idx} EN: {en}\n"
    last_err = None
    for _ in range(tries):
        try:
            out = subprocess.run(
                [
                    "pi",
                    "--model",
                    MODEL,
                    "-p",
                    "-nt",
                    "-nc",
                    "-ns",
                    "-ne",
                    "--no-session",
                    body,
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            return extract_json(out.stdout)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"GPT gen failed after {tries} tries: {last_err}")


def write_py(lang: str, trs: list[str]) -> None:
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
    (ROOT / "scripts" / "translations" / f"{lang}.py").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    lang = sys.argv[1]
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 75
    if lang not in NAMES:
        sys.exit(f"unknown language {lang}; expected one of {sorted(NAMES)}")
    bank = load_bank()
    if len(bank) != 460:
        sys.exit(f"expected 460 bank rows, got {len(bank)}")

    trs: list[str] = [""] * 460
    for i, row in enumerate(bank):
        if i in VERBATIM:
            trs[i] = row["text"]  # forced verbatim

    todo = [(i + 1, row["text"]) for i, row in enumerate(bank) if i not in VERBATIM]
    # batch over the todo list but keep original indices for placement
    for start in range(0, len(todo), batch):
        chunk = todo[start : start + batch]
        res = call_gpt(chunk, lang)
        rows = {r["i"]: r for r in res.get("rows", [])}
        for idx, _ in chunk:
            r = rows.get(idx)
            if r and r.get("text", "").strip():
                trs[idx - 1] = r["text"].strip()
            else:
                print(f"{lang} WARN row {idx} missing from model output", flush=True)
        print(
            f"{lang} batch {start + 1}-{start + len(chunk)}/{len(todo)} done",
            flush=True,
        )

    empty = [i + 1 for i, t in enumerate(trs) if not t.strip()]
    if empty:
        sys.exit(f"{lang}: rows still empty after generation: {empty}")

    write_py(lang, trs)
    print(f"{lang}: wrote scripts/translations/{lang}.py (460 rows)", flush=True)


if __name__ == "__main__":
    main()
