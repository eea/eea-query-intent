"""Author the English concept pools for the v1 exam and calibration slices.

Per docs/experiments/2026-09-17-iteration-v1-preregistration.md sections
5-6: fresh English short concepts (2-5 words) with concepts disjoint from
the 170-row training short bank (exam pool) and disjoint from each other
(exam vs calibration pool). Machine translation happens in a separate
pass (scripts/v1_translate_pools.py, next).

Outputs:
  data/pilot/v1-pools/en_exam_short.jsonl   90 rows (30/30/30)
  data/pilot/v1-pools/en_calib_short.jsonl  100 rows (70 retrieval, 15
  unknown, 15 eligible) — weighted toward hard negatives

Guards: 2-5 words, no casefold dupes within or across pools, no
collision with the frozen exam, exam pool has no collision with the
training bank.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "pilot" / "v1-pools"
EXAM = ROOT / "data" / "acceptance" / "v1" / "test.jsonl"
BANK = ROOT / "data" / "banks" / "v1-short" / "en.jsonl"

# ---------------------------------------------------------------- exam pool
EXAM_Q = [
    "why are bees declining?",
    "how do wetlands filter water?",
    "when did the WFD start?",
    "which birds eat microplastic?",
    "where do PFAS come from?",
    "do trees reduce air pollution?",
    "why are alpine glaciers shrinking?",
    "how fast is CO2 rising?",
    "can cities stop flooding?",
    "should peatlands be restored?",
    "what causes algal blooms?",
    "which metals poison rivers?",
    "how are dunes protected?",
    "why is the Danube drying?",
    "what kills coral reefs?",
    "when does smog peak?",
    "who pays for pollution?",
    "how loud is airport noise?",
    "which cars pollute most?",
    "does traffic cause asthma?",
    "why do lakes turn green?",
    "how are landfills capped?",
    "what heats cities most?",
    "where do microplastics accumulate?",
    "can wind farms hurt birds?",
    "why is Aral Sea shrinking?",
    "how are pesticides monitored?",
    "what drains groundwater fastest?",
    "which fuels cut emissions fastest?",
    "do green roofs cool homes?",
]
EXAM_E = [
    "tell me about algal blooms",
    "explain desertification in Europe",
    "more on peatland fires",
    "about glacial lake outbursts",
    "on invasive wasp species",
    "explain urban runoff",
    "more on noise pollution",
    "about light pollution effects",
    "on mercury in fish",
    "explain brownfield cleanup",
    "more on flood plains",
    "about e-waste recycling",
    "on sand mining rivers",
    "explain drought indexes",
    "more on heat waves",
    "about alpine permafrost thaw",
    "on microplastic in food",
    "explain carbon capture",
    "more on river rerouting",
    "about coastal erosion defenses",
    "on smog from industry",
    "explain soil sealing",
    "more on acid lakes",
    "about urban wildlife corridors",
    "on tire wear particles",
    "explain algal toxins",
    "more on heat islands",
    "about wetland carbon banks",
    "on construction noise rules",
    "explain nitrate leaching",
]
EXAM_C = [
    "peatlands store huge carbon",
    "cities should be greener",
    "fishing depletes river species",
    "noise harms heart health",
    "recycling saves raw materials",
    "agriculture drives nitrate runoff",
    "solar should be mandatory",
    "heat waves kill crops",
    "fly ash poisons soil",
    "electric cars cut smog",
    "dunes protect coastlines naturally",
    "the EU overregulates nature",
    "microplastics are in blood",
    "wind noise disturbs villages",
    "dams block salmon migration",
    "tourism strains island water",
    "grey water should be reused",
    "aviation dominates airport emissions",
    "beehives need pesticide reform",
    "landfills leak methane fast",
    "urban canals cool streets",
    "the green deal is weak",
    "plastic microbeads must vanish",
    "forests buffer storm surges",
    "gas plants delay net zero",
    "rivers carry most microplastic",
    "green taxes hit poor",
    "nature pays for floods",
    "electric shipping is coming",
    "the EU underfunds rivers",
]

# ------------------------------------------------------------ calibration pool
CALIB_R = [
    "co2 emissions europe 2024",
    "air quality index 2025",
    "wfd river basin maps",
    "ets price history",
    "natura 2000 site list",
    "pfas limits table",
    "lulucf report 2023",
    "green deal documents",
    "emissions by sector",
    "wildfire damage photos",
    "ozone hole size",
    "pm2.5 annual averages",
    "biodiversity strategy pdf",
    "climate law summary",
    "emissions trading rules",
    "water framework directive text",
    "natura 2000 map",
    "co2 map europe",
    "air pollution deaths",
    "plastic ban timeline",
    "flood risk zones",
    "land use map",
    "greenhouse gas inventory",
    "eea statistics 2024",
    "river discharge data",
    "soil contamination sites",
    "noise maps europe",
    "invasive species list",
    "protected areas list",
    "emissions reduction targets",
    "carbon budget calculator",
    "waste statistics europe",
    "water scarcity index",
    "heat wave alerts",
    "glacier retreat photos",
    "dune protection rules",
    "pesticide residues data",
    "marine litter survey",
    "urban sprawl maps",
    "renewable energy share",
    "transport emissions split",
    "building emissions share",
    "agriculture emissions share",
    "landfill gas statistics",
    "drinking water quality",
    "bathing water results",
    "ground water levels",
    "air quality alerts",
    "smog episodes 2024",
    "carbon price forecast",
    "ets cap levels",
    "emissions certificates price",
    "co2 border tax",
    "reforestation budget 2024",
    "nature restoration law",
    "peatland map europe",
    "wildlife corridor projects",
    "flood protection costs",
    "drought index europe",
    "air pollution map",
    "water usage statistics",
    "energy consumption trends",
    "co2 per capita",
    "methane emissions agriculture",
    "nitrate levels map",
    "phosphorus runoff data",
    "algal bloom alerts",
    "dead zones map",
    "ocean temperature records",
    "sea level rise",
]
CALIB_U = [
    "football transfer news",
    "carbonara recipe pasta",
    "chess opening book",
    "stock market crash 2008",
    "learn spanish fast",
    "dog training tricks",
    "iphone repair guide",
    "movie streaming service",
    "crypto coin prices",
    "yoga poses list",
    "wedding dress ideas",
    "car tire pressure",
    "basketball finals scores",
    "dream about flying",
    "guitar chord charts",
]
CALIB_Q = [
    "why do cliffs collapse?",
    "how is waste water cleaned?",
    "when do rivers freeze?",
    "can oysters clean rivers?",
    "which fish have mercury?",
]
CALIB_E = [
    "about tidal energy",
    "on industrial dust",
    "explain river deltas",
    "more on e-waste",
    "about coastal floods",
]
CALIB_C = [
    "harbours block fish spawns",
    "water bills rise too fast",
    "algae blooms choke rivers",
    "fish stocks recover fast",
    "sea levels swallow coasts",
]


def check(texts: list[str], where: str) -> None:
    seen: set[str] = set()
    for t in texts:
        w = len(t.split())
        assert 2 <= w <= 5, f"{where}: bad length ({w}): {t!r}"
        f = t.casefold().strip()
        assert f not in seen, f"{where}: duplicate: {t!r}"
        seen.add(f)
    return seen


def write_pool(path: Path, rows: list[tuple[str, str]]) -> None:
    prefix = "v1ex" if "exam" in path.name else "v1cl"
    source = "v1_exam_authored" if prefix == "v1ex" else "v1_calib_authored"
    with path.open("w", encoding="utf-8") as fh:
        for n, (intent, text) in enumerate(rows, 1):
            rec = {
                "id": f"{prefix}-en-{intent}-{n:04d}",
                "intent": intent,
                "language": "en",
                "review_status": "manual",
                "source_type": source,
                "split": "train",
                "template_id": f"{prefix}-en-{intent}-{n:04d}",
                "text": text,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    exam_rows = (
        [("question", t) for t in EXAM_Q]
        + [("exploratory", t) for t in EXAM_E]
        + [("claim", t) for t in EXAM_C]
    )
    calib_rows = (
        [("retrieval", t) for t in CALIB_R]
        + [("unknown", t) for t in CALIB_U]
        + [("question", t) for t in CALIB_Q]
        + [("exploratory", t) for t in CALIB_E]
        + [("claim", t) for t in CALIB_C]
    )
    assert len(exam_rows) == 90, len(exam_rows)
    assert len(calib_rows) == 100, len(calib_rows)

    exam_seen = check([t for _, t in exam_rows], "exam")
    calib_seen = check([t for _, t in calib_rows], "calib")
    overlap = exam_seen & calib_seen
    assert not overlap, f"exam/calib concept overlap: {overlap}"

    exam_frozen = {
        json.loads(line)["text"].casefold()
        for line in EXAM.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    hit = exam_seen & exam_frozen
    assert not hit, f"exam pool collides with frozen exam: {hit}"

    bank_seen = {
        json.loads(line)["text"].casefold()
        for line in BANK.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    hit = exam_seen & bank_seen
    assert not hit, f"exam pool collides with training bank: {hit}"

    write_pool(OUT / "en_exam_short.jsonl", exam_rows)
    write_pool(OUT / "en_calib_short.jsonl", calib_rows)
    print(f"exam pool:   {len(exam_rows)} concepts -> {OUT / 'en_exam_short.jsonl'}")
    print(f"calib pool:  {len(calib_rows)} concepts -> {OUT / 'en_calib_short.jsonl'}")


if __name__ == "__main__":
    main()
