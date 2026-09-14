"""Replace acceptance rows that overlap (case-insensitive) with the existing
training / regression corpus, so the holdout stays honest.

For one language: find every shard row whose text already exists in
``data/expanded_v2``, ``data/english_only``, ``data/seed``, or
``data/multilingual/v1``; ask GPT-5.6 Sol (one call) for a different query of
the same intent for each collision, with the colliding strings and similar
existing strings listed as forbidden; verify locally (no overlap, no
in-shard duplicate, <= 20 words) and rewrite the shard in place (same id
slot, new text).

Usage: uv run python scripts/repair_acceptance_overlaps.py <lang>
"""

import json
import sys
from pathlib import Path

from gpt_acceptance_gen import NAMES, QA_MODEL, call_gpt_raw

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "acceptance" / "v1"

EXISTING_FILES = (
    "data/expanded_v2/train.jsonl",
    "data/expanded_v2/calibration.jsonl",
    "data/english_only/train.jsonl",
    "data/english_only/calibration.jsonl",
    "data/seed/english.jsonl",
    "data/multilingual/v1/test.jsonl",
)


def training_finals() -> tuple[str, ...]:
    d = ROOT / "data" / "training" / "v1"
    return tuple(
        str(p.relative_to(ROOT))
        for p in sorted(d.glob("*.jsonl"))
        if not p.name.endswith(".raw.jsonl")
    )


PROMPT_TMPL = """You are a native @NAME@ speaker. We are building a held-out evaluation corpus of search-box queries for the European Environment Agency website, and the following @N@ queries (each with its intent label) collide with strings that were already used in the model's training data. For each one, generate ONE replacement query with the same intent and similar shape, in @NAME@.

Rules:
- The replacement must NOT be identical to, or a trivial variant of, any string in the FORBIDDEN list (which includes similar existing queries).
- Same intent: question = complete natural-language question; exploratory = explain/compare/overview/trend request; claim = complete declarative sentence asserting a checkable environmental fact; retrieval = short noun phrase / keyword / document / map / data lookup (1-6 words, NOT a full sentence); unknown = code-switched fragment, gibberish, or out-of-scope non-environmental query.
- Keep the search-box register (terse, realistic), at most 20 words, correct native grammar and diacritics.
- Acronyms and proper nouns (PM2.5, CO2, NO2, SOER, EU, EEA, city names) may stay Latin.

Output STRICT JSON only: {{"rows":[{{"i":1,"text":"<replacement>"}}]}} - one entry per numbered row, in order.

Rows:
"""


def load_existing() -> set[str]:
    texts: set[str] = set()
    for rel in EXISTING_FILES + training_finals():
        path = ROOT / rel
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                texts.add(json.loads(line)["text"].casefold())
    return texts


def similar_forbid(text: str, existing: set[str], limit: int = 120) -> list[str]:
    first = text.split()[0].casefold() if text.split() else ""
    out = [t for t in existing if t.split() and t.split()[0] == first]
    return sorted(out)[:limit]


def main() -> None:
    lang = sys.argv[1]
    shard = DATA_DIR / f"{lang}.jsonl"
    rows = [
        json.loads(line)
        for line in shard.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    existing = load_existing()
    shard_texts = {r["text"].casefold() for r in rows}

    collisions = [idx for idx, r in enumerate(rows) if r["text"].casefold() in existing]
    if not collisions:
        print(f"{lang}: no overlaps, nothing to repair", flush=True)
        return

    print(f"{lang}: {len(collisions)} overlapping rows to repair", flush=True)

    for _round in range(3):
        prompt = PROMPT_TMPL.replace("@N@", str(len(collisions))).replace(
            "@NAME@", NAMES[lang]
        )
        for i, idx in enumerate(collisions, start=1):
            row = rows[idx]
            forbid = {row["text"].casefold()}
            for t in similar_forbid(row["text"], existing):
                forbid.add(t)
            forbid_text = "\n".join(f"- {t}" for t in sorted(forbid))
            prompt += f"{i} [{row['intent']}] {row['text']}\nFORBIDDEN (similar existing strings):\n{forbid_text}\n\n"
        res = call_gpt_raw(prompt, model=QA_MODEL)
        by_i = {
            r["i"]: r["text"].strip()
            for r in res.get("rows", [])
            if r.get("text", "").strip()
        }

        still_bad: list[int] = []
        for i, idx in enumerate(collisions, start=1):
            cand = by_i.get(i, "")
            if not cand or len(cand.split()) > 20:
                still_bad.append(idx)
                continue
            key = cand.casefold()
            if key in existing or key in shard_texts:
                still_bad.append(idx)
                continue
            shard_texts.discard(rows[idx]["text"].casefold())
            rows[idx]["text"] = cand
            shard_texts.add(key)
        collisions = still_bad
        print(
            f"{lang}: round {_round + 1} -> {len(collisions)} still colliding",
            flush=True,
        )
        if not collisions:
            break
    else:
        sys.exit(f"{lang}: {len(collisions)} rows still collide after 3 rounds")

    with shard.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{lang}: shard rewritten, overlaps repaired", flush=True)


if __name__ == "__main__":
    main()
