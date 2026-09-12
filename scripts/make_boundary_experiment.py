"""Build a deterministic topic+location boundary dataset for the English
validation experiment (no GPT involved).

The English acceptance measurement showed the model's false AI routes are
almost all "topic + location" retrieval phrases (e.g. "floods in Germany",
p~0.99). This script hammers exactly that boundary:

- RETRIEVAL rows: "topic location" noun phrases from the EEA topic pool x
  country/region pool (the no-AI shape the model misroutes).
- QUESTION rows: the SAME topic+location pairs wrapped in grammatically
  safe question templates (the eligible shape), so the model sees the
  direct contrast: same content, keyword form -> no AI, full question -> AI.

Every row is deduplicated case-insensitively against the acceptance holdout
and all existing datasets. Output: data/en_experiment/boundary.jsonl.

Usage: uv run python scripts/make_boundary_experiment.py
"""

import json
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TOPICS = (
    "air pollution",
    "PM2.5",
    "PM10",
    "nitrogen dioxide",
    "ozone",
    "sulphur dioxide",
    "ammonia",
    "greenhouse gas emissions",
    "CO2",
    "climate neutrality",
    "the European Green Deal",
    "Fit for 55",
    "the Emissions Trading System",
    "LULUCF",
    "renewable energy",
    "the Water Framework Directive",
    "bathing waters",
    "European water quality",
    "groundwater",
    "floods",
    "droughts",
    "biodiversity",
    "Natura 2000",
    "protected areas",
    "forests",
    "deforestation",
    "soil",
    "pollinators",
    "waste",
    "recycling",
    "plastics",
    "microplastics",
    "e-waste",
    "landfill",
    "the circular economy",
    "PFAS",
    "heavy metals",
    "pesticides",
    "persistent organic pollutants",
    "noise",
    "urban heat islands",
    "the ozone layer",
    "transport emissions",
    "aviation emissions",
    "agriculture",
    "nitrogen",
    "health effects of pollution",
)

LOCATIONS = (
    "Romania",
    "Germany",
    "France",
    "Poland",
    "Italy",
    "Spain",
    "Greece",
    "Hungary",
    "Czechia",
    "Sweden",
    "Austria",
    "the Netherlands",
    "Belgium",
    "Portugal",
    "Ireland",
    "Finland",
    "Denmark",
    "Croatia",
    "Slovenia",
    "Slovakia",
    "Bulgaria",
    "Lithuania",
    "Latvia",
    "Estonia",
    "Malta",
    "Cyprus",
    "Luxembourg",
    "Europe",
    "the EU",
    "the Baltic Sea",
    "the Mediterranean",
    "the Black Sea",
    "the North Sea",
    "the Rhine",
    "the Danube",
    "the Alps",
    "the Western Balkans",
)

QUESTION_TEMPLATES = (
    "What is the current state of {topic} in {location}?",
    "What does the EEA say about {topic} in {location}?",
    "Is {topic} getting worse or better in {location}?",
    "How has {topic} changed in {location} over the last decade?",
)

DEDUP_DIRS = [
    ROOT / "data" / "acceptance" / "v1",
    ROOT / "data" / "expanded_v2",
    ROOT / "data" / "english_only",
    ROOT / "data" / "seed",
    ROOT / "data" / "multilingual" / "v1",
]

MAX_RETRIEVAL = 2000
MAX_QUESTIONS = 2000


def existing_texts() -> set[str]:
    seen = set()
    for d in DEDUP_DIRS:
        if not d.exists():
            continue
        for path in d.glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    seen.add(json.loads(line)["text"].casefold())
    return seen


def main() -> None:
    avoid = existing_texts()
    out_dir = ROOT / "data" / "en_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    # pair list: every topic x location, deduped against all datasets
    pairs: list[tuple[str, str]] = []
    for topic, location in product(TOPICS, LOCATIONS):
        text = f"{topic} {location}".strip()
        if text.casefold() in avoid:
            continue
        avoid.add(text.casefold())
        pairs.append((topic, location))
    print(f"topic+location pairs kept: {len(pairs)}")

    records: list[dict] = []

    # retrieval rows: the no-AI keyword shape
    for seq, (topic, location) in enumerate(pairs[:MAX_RETRIEVAL], start=1):
        text = f"{topic} {location}".strip()
        row_id = f"bnd-en-ret-{seq:04d}"
        records.append(
            {
                "id": row_id,
                "template_id": row_id,
                "language": "en",
                "text": text,
                "intent": "retrieval",
                "source_type": "synthetic_generated",
                "review_status": "policy_reviewed",
                "split": "train",
            }
        )

    # question rows: same pairs, full-question form (direct contrast)
    used: set[str] = set()
    q_count = 0
    for _seq, (topic, location) in enumerate(pairs, start=1):
        for _t_idx, template in enumerate(QUESTION_TEMPLATES, start=1):
            text = template.format(topic=topic, location=location)
            key = text.casefold()
            if key in used or key in avoid:
                continue
            used.add(key)
            q_count += 1
            row_id = f"bnd-en-qst-{q_count:05d}"
            records.append(
                {
                    "id": row_id,
                    "template_id": row_id,
                    "language": "en",
                    "text": text,
                    "intent": "question",
                    "source_type": "synthetic_generated",
                    "review_status": "policy_reviewed",
                    "split": "train",
                }
            )
            if q_count >= MAX_QUESTIONS:
                break
        if q_count >= MAX_QUESTIONS:
            break

    out_path = out_dir / "boundary.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_ret = sum(1 for r in records if r["intent"] == "retrieval")
    n_q = sum(1 for r in records if r["intent"] == "question")
    print(f"wrote {out_path}: {n_ret} retrieval + {n_q} question = {len(records)} rows")


if __name__ == "__main__":
    main()
