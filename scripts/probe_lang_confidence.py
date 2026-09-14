"""Per-language confidence probe for the 27-language training scale-up.

For every non-English language, the in-house model (Gemma 4 31B IT)
generates a small sample of question rows - full sentences, the highest
grammar-risk intent. A Codex model (GPT-5.6 Luna) then judges each row
as a native speaker of that language.

Routing rule (user decision 2026-09-13): use GPT where the in-house
model is not confident about a language. Operationalized as: if the
measured flaw rate in the sample is at or above FLAW_THRESHOLD, that
language's full training corpus (generation + QA + top-up) runs on GPT
Luna; otherwise it stays on the free in-house gateway. A probe that
cannot be measured (transport failure) routes to GPT as well - the
uncertain direction is the safe one.

Outputs:
  data/train_routing.json             routing table (committed)
  .pipeline/train_routing.sh          shell-readable vars for workers
                                      (not a .env name: the secrets guard
                                      blocks reads of *.env files)
  reports/lang_confidence_probe.json  full per-row results (gitignored)
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from gpt_acceptance_gen import (  # noqa: E402
    NAMES,
    build_prompt,
    call_gpt,
    call_gpt_raw,
)

try:
    from eea_query_intent.languages import SUPPORTED_LANGUAGE_CODES
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(ROOT / "src"))
    from eea_query_intent.languages import SUPPORTED_LANGUAGE_CODES

INHOUSE = "EEA/Inhouse-LLM/gemma-4-31B-it"
JUDGE = "openai-codex/gpt-5.6-luna"
GPT = "openai-codex/gpt-5.6-luna"
PROBE_ROWS = 25
FLAW_THRESHOLD = 0.25
PARALLEL = 4

JUDGE_TMPL = """You are a native speaker of @LANG@ proofreading search queries.
Below are @N@ search queries written in @LANG@ (numbered, one per line).

A query is OK if it is natural, grammatical @LANG@ in a short search register.
A query is FLAWED if it has grammar, spelling, or word-formation errors;
unnatural code-switching; or if it is not really @LANG@ at all.
Proper nouns and technical terms (PFAS, PM2.5, SOER, CO2, NO2, EU, EEA,
country names) stay as-is and are never flaws. Judge language quality only,
not the topic.

Queries:
@ROWS@

Reply with STRICT JSON only, one entry per query in order:
{"rows": [{"i": 1, "ok": true}, {"i": 2, "ok": false}]}"""


def judge(rows: list[str], lang: str) -> list[bool]:
    numbered = "\n".join(f"{i}. {r}" for i, r in enumerate(rows, 1))
    prompt = (
        JUDGE_TMPL.replace("@LANG@", NAMES[lang])
        .replace("@N@", str(len(rows)))
        .replace("@ROWS@", numbered)
    )
    res = call_gpt_raw(prompt, tries=3, model=JUDGE)
    verdicts = {}
    for item in res.get("rows", []):
        try:
            verdicts[int(item["i"])] = bool(item.get("ok", False))
        except (KeyError, TypeError, ValueError):
            continue
    # missing verdicts count as flawed (fail safe toward GPT)
    return [verdicts.get(i, False) for i in range(1, len(rows) + 1)]


def probe_one(lang: str) -> dict:
    rows = call_gpt(build_prompt(lang, "question", PROBE_ROWS), model=INHOUSE)
    rows = [r.strip() for r in rows if r.strip()][:PROBE_ROWS]
    if not rows:
        return {"lang": lang, "error": "no rows generated", "flaw_rate": 1.0}
    verdicts = judge(rows, lang)
    flawed = sum(1 for v in verdicts if not v)
    rate = flawed / len(rows)
    routed = "gpt" if rate >= FLAW_THRESHOLD else "inhouse"
    return {
        "lang": lang,
        "sample_size": len(rows),
        "flawed": flawed,
        "flaw_rate": round(rate, 3),
        "routed": routed,
        "rows": [{"text": t, "ok": v} for t, v in zip(rows, verdicts, strict=True)],
    }


def main() -> None:
    langs = sorted(SUPPORTED_LANGUAGE_CODES - {"en"})
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        futures = {pool.submit(probe_one, lang): lang for lang in langs}
        for future in as_completed(futures):
            lang = futures[future]
            try:
                results[lang] = future.result()
            except Exception as e:  # noqa: BLE001
                results[lang] = {
                    "lang": lang,
                    "error": str(e),
                    "flaw_rate": 1.0,
                    "routed": "gpt",
                }
            rec = results[lang]
            print(
                f"{lang}: flaw_rate={rec.get('flaw_rate')} "
                f"routed={rec.get('routed')} "
                f"{rec.get('error', '')}",
                flush=True,
            )

    routing = {
        lang: {
            "gen_model": GPT if results[lang]["routed"] == "gpt" else INHOUSE,
            "qa_model": GPT if results[lang]["routed"] == "gpt" else INHOUSE,
            "flaw_rate": results[lang].get("flaw_rate"),
            "reason": results[lang].get("error", "probe"),
        }
        for lang in langs
    }

    (ROOT / "data").mkdir(exist_ok=True)
    with (ROOT / "data" / "train_routing.json").open("w", encoding="utf-8") as h:
        json.dump(routing, h, ensure_ascii=False, indent=2)
        h.write("\n")

    env_lines = ["# generated by scripts/probe_lang_confidence.py", ""]
    for lang in langs:
        env_lines.append(f"TRAIN_GEN_{lang}={routing[lang]['gen_model']}")
        env_lines.append(f"TRAIN_QA_{lang}={routing[lang]['qa_model']}")
    (ROOT / ".pipeline").mkdir(exist_ok=True)
    (ROOT / ".pipeline" / "train_routing.sh").write_text(
        "\n".join(env_lines) + "\n", encoding="utf-8"
    )

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    with (reports / "lang_confidence_probe.json").open("w", encoding="utf-8") as h:
        json.dump(
            {
                "threshold": FLAW_THRESHOLD,
                "probe_rows": PROBE_ROWS,
                "judge": JUDGE,
                "generator": INHOUSE,
                "results": results,
            },
            h,
            ensure_ascii=False,
            indent=2,
        )
        h.write("\n")

    gpt_langs = [lang for lang in langs if results[lang]["routed"] == "gpt"]
    print(
        f"\nGPT-routed ({len(gpt_langs)}): {' '.join(gpt_langs) or 'none'}\n"
        f"in-house ({len(langs) - len(gpt_langs)}): "
        f"{' '.join(lang for lang in langs if results[lang]['routed'] != 'gpt')}"
    )


if __name__ == "__main__":
    main()
