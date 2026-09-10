"""Generate the multilingual v1 dataset from the template and topic banks.

Design rules:
- A split is assigned per *template* (by hash of its id), so every
  translation of the same semantic template lands in the same split. This is
  the structural defence against translation leakage (see annotation spec).
- Slot filling is deterministic: a small number of topic/year fills per
  template, chosen by hash, so regeneration is stable.
- Output: one JSONL file per split under data/multilingual/v1/.
- The English seed corpus (policy-reviewed rows from the frontend) is added
  to train.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_DIR = DATA / "multilingual" / "v1"

YEARS = ["2019", "2020", "2022", "2024", "2025", "2030"]
TOPICS_PER_FILL = 4
YEARS_PER_FILL = 2
SPLITS = ("train", "validation", "calibration", "test")
INTENT_TAG = {
    "questions": "q",
    "exploratory": "e",
    "claims": "c",
    "retrieval": "r",
    "unknown": "u",
}


def digest(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest(), 16)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def fill(
    text: str, template_id: str, language: str, variant: int, topics: list[dict]
) -> str:
    def topic_value(capitalized: bool) -> str:
        index = (digest(template_id, language) + variant * 7) % len(topics)
        value = topics[index]["texts"][language]
        if capitalized:
            value = value[0].upper() + value[1:]
        return value

    def year_value() -> str:
        return YEARS[(digest(template_id, language) + variant * 3) % len(YEARS)]

    text = re.sub(r"\{Topic\}", lambda _m: topic_value(True), text)
    text = re.sub(r"\{topic\}", lambda _m: topic_value(False), text)
    text = re.sub(r"\{year\}", lambda _m: year_value(), text)
    return text


def main() -> int:
    topics = load_jsonl(DATA / "topics.jsonl")
    assert len(topics) == 14, f"expected 14 topics, got {len(topics)}"

    template_files = {
        "questions": DATA / "templates" / "questions.jsonl",
        "exploratory": DATA / "templates" / "exploratory.jsonl",
        "claims": DATA / "templates" / "claims.jsonl",
        "retrieval": DATA / "templates" / "retrieval.jsonl",
        "unknown": DATA / "templates" / "unknown.jsonl",
    }

    # Balanced split assignment: cycle a 7:1:1:1 pattern over the sorted
    # template ids so every split gets an even share of templates (and all
    # of a template's translations).
    all_ids = sorted(
        template["id"]
        for path in template_files.values()
        for template in load_jsonl(path)
    )
    pattern = ["train"] * 7 + ["validation", "calibration", "test"]
    split_by_template = {
        template_id: pattern[index % len(pattern)]
        for index, template_id in enumerate(all_ids)
    }

    rows_by_split: dict[str, list[dict]] = {s: [] for s in SPLITS}

    total = 0
    for intent_name, path in template_files.items():
        intent = {
            "questions": "question",
            "exploratory": "exploratory",
            "claims": "claim",
            "retrieval": "retrieval",
            "unknown": "unknown",
        }[intent_name]
        tag = INTENT_TAG[intent_name]
        for template in load_jsonl(path):
            template_id = template["id"]
            text_template = template["texts"]
            slots = template.get("slots", [])
            split = split_by_template[template_id]
            n_fills = (
                TOPICS_PER_FILL
                if "topic" in slots
                else (YEARS_PER_FILL if "year" in slots else 1)
            )
            for language, lang_template in text_template.items():
                if not lang_template:
                    continue
                for variant in range(n_fills):
                    text = fill(lang_template, template_id, language, variant, topics)
                    row = {
                        "id": f"{language}-{tag}-{template_id}-{variant:02d}",
                        "language": language,
                        "text": text,
                        "intent": intent,
                        "template_id": f"template-{template_id}",
                        "source_type": (
                            "synthetic_generated"
                            if language == "en"
                            else "synthetic_translated"
                        ),
                        "review_status": "policy_reviewed",
                        "split": split,
                    }
                    rows_by_split[split].append(row)
                    total += 1

    # Fold the English policy seed corpus into train.
    seed_path = DATA / "seed" / "english.jsonl"
    seed_rows = load_jsonl(seed_path)
    for row in seed_rows:
        row = dict(row)
        row["split"] = "train"
        rows_by_split["train"].append(row)
        total += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = OUT_DIR / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows_by_split[split]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows_by_split[split])} rows")
    print(f"total: {total} rows")

    # Self-validation: reload everything and run the dataset validator,
    # which enforces the cross-split template-leakage rule.
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        sys.path.insert(0, str(ROOT / "src"))
        from eea_query_intent.dataset import load_dataset  # noqa: E402

        combined = OUT_DIR / "combined.jsonl"
        with combined.open("w", encoding="utf-8") as handle:
            for split in SPLITS:
                for line in (
                    (OUT_DIR / f"{split}.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ):
                    handle.write(line + "\n")
        records = load_dataset(combined)
        print(f"validator OK: {len(records)} records, no leakage")
        combined.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
