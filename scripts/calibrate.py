"""Calibrate the abstention threshold for a candidate model.

Sweeps the eligible-probability threshold on the calibration split and
picks the smallest threshold whose worst-language no-AI false-positive
rate is at or below the target. If no threshold reaches the target, the
threshold minimizing the worst-language rate is chosen and reported.

The chosen threshold is written into the model manifest so the service
and the evaluation harness use the same value.

Usage:
    uv run python scripts/calibrate.py setfit
"""

from __future__ import annotations

import json
from pathlib import Path

from eea_query_intent.contracts import AI_ELIGIBLE_INTENTS

ROOT = Path(__file__).resolve().parent.parent
CALIBRATION = ROOT / "data/expanded_v2" / "calibration.jsonl"
THRESHOLDS = [round(0.50 + 0.01 * i, 2) for i in range(50)]  # 0.50 .. 0.99
TARGET_WORST_FP_RATE = 0.01


def load_calibration() -> list[dict]:
    return [
        json.loads(line)
        for line in CALIBRATION.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_predictions(model_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in (
        (model_dir / "calibration-predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        if not line.strip():
            continue
        prediction = json.loads(line)
        out[prediction["id"]] = prediction["eligible_probability"]
    return out


def evaluate_threshold(
    gold: list[dict],
    probabilities: dict[str, float],
    threshold: float,
) -> tuple[float, float, int]:
    """Return (worst-language no-AI FP rate, eligible miss rate, misses)."""
    per_lang: dict[str, dict[str, int]] = {}
    total_eligible = 0
    total_missed = 0
    for row in gold:
        gold_eligible = row["intent"] in AI_ELIGIBLE_INTENTS
        predicted_eligible = probabilities[row["id"]] >= threshold
        bucket = per_lang.setdefault(
            row["language"],
            {"no_ai": 0, "false_positive": 0},
        )
        if gold_eligible:
            total_eligible += 1
            if not predicted_eligible:
                total_missed += 1
        else:
            bucket["no_ai"] += 1
            if predicted_eligible:
                bucket["false_positive"] += 1
    worst = 0.0
    for bucket in per_lang.values():
        if bucket["no_ai"]:
            worst = max(worst, bucket["false_positive"] / bucket["no_ai"])
    missed_rate = total_missed / total_eligible if total_eligible else 0.0
    return worst, missed_rate, total_missed


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_type", choices=["setfit"])
    parser.add_argument(
        "--model-dir",
        default=None,
        help="model dir (default: models/<model_type>)",
    )
    parser.add_argument(
        "--calibration-file",
        default=str(CALIBRATION),
        help="calibration gold split",
    )
    args = parser.parse_args()

    model_dir = (
        Path(args.model_dir) if args.model_dir else ROOT / "models" / args.model_type
    )

    gold = [
        json.loads(line)
        for line in Path(args.calibration_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    probabilities = load_predictions(model_dir)
    missing = [row["id"] for row in gold if row["id"] not in probabilities]
    if missing:
        raise SystemExit(f"missing predictions for {len(missing)} rows")

    results = []
    for threshold in THRESHOLDS:
        worst, missed_rate, missed = evaluate_threshold(gold, probabilities, threshold)
        results.append(
            {
                "threshold": threshold,
                "worst_language_no_ai_fp_rate": worst,
                "eligible_miss_rate": missed_rate,
                "eligible_misses": missed,
            }
        )

    meeting = [
        r for r in results if r["worst_language_no_ai_fp_rate"] <= TARGET_WORST_FP_RATE
    ]
    if meeting:
        chosen = meeting[0]
        selection = "smallest threshold meeting the worst-language target"
    else:
        chosen = min(results, key=lambda r: r["worst_language_no_ai_fp_rate"])
        selection = "no threshold meets the target; minimum worst-language rate"

    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["abstain_threshold"] = chosen["threshold"]
    manifest["calibration"] = {
        "split": args.calibration_file,
        "records": len(gold),
        "selection_rule": selection,
        "target_worst_no_ai_fp_rate": TARGET_WORST_FP_RATE,
        "chosen": chosen,
        "sweep": results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"{args.model_type}: threshold={chosen['threshold']} "
        f"({selection}) "
        f"worst-language FP rate={chosen['worst_language_no_ai_fp_rate']:.3f} "
        f"eligible miss rate={chosen['eligible_miss_rate']:.3f}"
    )
    print(f"manifest updated: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
