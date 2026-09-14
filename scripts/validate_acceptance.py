"""Validate the acceptance corpus and (optionally) build the merged test file.

Checks, per language shard in ``data/acceptance/v1/<lang>.jsonl``:

1. row integrity: non-empty, <= 20 words, no intra-file duplicate
   (case-insensitive), per-intent counts equal the acceptance targets;
2. no case-insensitive text overlap with any training / calibration /
   regression file (``data/expanded_v2``, ``data/english_only``,
   ``data/seed``, ``data/multilingual/v1``);
3. per-language character-set sanity (native script + Latin acronyms);
4. template_id uniqueness inside the corpus and zero collision with the
   training template ids (translation/template leakage guard);
5. metadata: review_status llm_reviewed, split test, language matches shard.

With ``--merge`` a passing run also writes ``data/acceptance/v1/test.jsonl``
(sorted by id).

Usage: uv run python scripts/validate_acceptance.py [--merge]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "acceptance" / "v1"

TARGETS = {
    "question": 150,
    "exploratory": 120,
    "claim": 90,
    "retrieval": 350,
    "unknown": 50,
}
INTENTS = ("question", "exploratory", "claim", "retrieval", "unknown")

EXISTING_FILES = (
    "data/expanded_v2/train.jsonl",
    "data/expanded_v2/calibration.jsonl",
    "data/english_only/train.jsonl",
    "data/english_only/calibration.jsonl",
    "data/seed/english.jsonl",
    "data/multilingual/v1/test.jsonl",
)
# plus every per-language training corpus final (the 3000-row corpora in
# data/training/v1) - discovered dynamically so new languages are covered
def training_finals() -> tuple[str, ...]:
    d = ROOT / "data" / "training" / "v1"
    return tuple(
        str(p.relative_to(ROOT))
        for p in sorted(d.glob("*.jsonl"))
        if not p.name.endswith(".raw.jsonl")
    )

# Per-language allowed letter ranges (besides ASCII letters/digits and the
# small Unicode punctuation allow-list).
CYRILLIC = frozenset(range(0x0400, 0x0500))
GREEK = frozenset(range(0x0370, 0x0400))
LATIN = (
    frozenset(range(0x0041, 0x005B))
    | frozenset(range(0x0061, 0x007B))
    # Latin-1 punctuation/symbols (¿ ¡ ° ± µ · …) + Latin Extended
    | frozenset(range(0x00A0, 0x0250))
    # spacing modifier letters (standalone diacritics such as ˇ ˚ ˛)
    | frozenset(range(0x02B0, 0x0300))
    # combining diacritical marks (decomposed sequences)
    | frozenset(range(0x0300, 0x0370))
)
PUNCT = (
    frozenset(
        range(0x0020, 0x007F)  # ASCII (letters handled separately)
    )
    | frozenset(range(0x2010, 0x205F))  # general punctuation
)


def allowed_ranges(lang: str) -> frozenset:
    if lang == "bg":
        return CYRILLIC
    if lang == "el":
        return GREEK
    return LATIN


def text_ok(text: str, lang: str) -> str | None:
    if not text or not text.strip():
        return "empty"
    if len(text.split()) > 20:
        return "too_long"
    allowed = allowed_ranges(lang)
    for ch in text:
        code = ord(ch)
        if code in PUNCT or code in allowed or ch.isascii() and ch.isalnum():
            continue
        return f"unexpected character {ch!r} (U+{code:04X})"
    return None


def load_existing() -> tuple[set[str], set[str]]:
    texts: set[str] = set()
    template_ids: set[str] = set()
    for rel in EXISTING_FILES:
        path = ROOT / rel
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            texts.add(row["text"].casefold())
            if row.get("template_id"):
                template_ids.add(row["template_id"])
    return texts, template_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge", action="store_true", help="write merged test.jsonl")
    args = parser.parse_args()

    shards = sorted(
        p
        for p in DATA_DIR.glob("*.jsonl")
        if not p.name.endswith(".raw.jsonl") and p.stem != "test"
    )
    if not shards:
        sys.exit("no acceptance shards found in data/acceptance/v1")

    global EXISTING_FILES
    EXISTING_FILES = EXISTING_FILES + training_finals()
    existing_texts, existing_templates = load_existing()
    errors: list[str] = []
    all_rows: list[dict] = []
    seen_templates: set[str] = set()
    per_language: dict[str, dict] = {}

    for shard in shards:
        lang = shard.stem
        rows = [
            json.loads(line)
            for line in shard.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        counts = {intent: 0 for intent in INTENTS}
        seen_texts: set[str] = set()
        for row in rows:
            text = row["text"]
            if row.get("language") != lang:
                errors.append(
                    f"{lang}: row {row.get('id')} has language {row.get('language')}"
                )
            if row.get("intent") not in INTENTS:
                errors.append(
                    f"{lang}: row {row.get('id')} bad intent {row.get('intent')}"
                )
                continue
            counts[row["intent"]] += 1
            problem = text_ok(text, lang)
            if problem:
                errors.append(f"{lang}: row {row.get('id')}: {problem}")
            key = text.casefold()
            if key in seen_texts:
                errors.append(f"{lang}: duplicate text {text!r}")
            seen_texts.add(key)
            if key in existing_texts:
                errors.append(
                    f"{lang}: row {row.get('id')} overlaps existing data: {text!r}"
                )
            tid = row.get("template_id")
            if tid in seen_templates:
                errors.append(f"{lang}: duplicate template_id {tid}")
            seen_templates.add(tid)
            if tid in existing_templates:
                errors.append(f"{lang}: template_id {tid} collides with training data")
            if row.get("review_status") != "llm_reviewed":
                errors.append(f"{lang}: row {row.get('id')} not llm_reviewed")
            if row.get("split") != "test":
                errors.append(f"{lang}: row {row.get('id')} split != test")
            all_rows.append(row)
        for intent in INTENTS:
            if counts[intent] != TARGETS[intent]:
                errors.append(
                    f"{lang}: {intent} count {counts[intent]} != {TARGETS[intent]}"
                )
        per_language[lang] = {"rows": len(rows), "counts": counts}

    summary = {
        "languages": len(shards),
        "records": len(all_rows),
        "per_language": per_language,
        "errors": errors,
        "passed": not errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if errors:
        for err in errors[:20]:
            print(f"ERROR {err}", file=sys.stderr)
        if len(errors) > 20:
            print(f"ERROR ... and {len(errors) - 20} more", file=sys.stderr)
        sys.exit(1)

    if args.merge:
        merged = sorted(all_rows, key=lambda r: r["id"])
        out = DATA_DIR / "test.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for row in merged:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"merged {len(merged)} rows -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
