"""Sweep abstain thresholds over acceptance predictions.

The prediction file carries the raw eligible_probability; this script
re-gates it at each candidate threshold (mirroring the service rule:
routed = p_eligible >= threshold) and runs the evaluate gate, printing a
per-threshold table of worst-language no-AI false-positive rate, average
eligible recall, and abstention. This is the threshold-selection tool for
the final model once the full 28-language acceptance set exists.

Usage:
  uv run python scripts/sweep_acceptance.py \
    --gold data/acceptance/v1/test.jsonl \
    --preds models/setfit/acceptance-predictions.jsonl \
    --thresholds 0.90,0.92,0.94,0.95,0.96,0.97,0.98,0.99,0.995
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ELIGIBLE = {"question", "exploratory", "claim"}


def regate(preds_path: Path, threshold: float) -> list[dict]:
    out = []
    with preds_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            p = rec["eligible_probability"]
            if p < threshold:
                rec["intent"] = "unknown"
                rec["eligible"] = False
                rec["abstained"] = True
            else:
                rec["intent"] = rec.get("raw_intent", rec["intent"])
                rec["eligible"] = rec["intent"] in ELIGIBLE
                rec["abstained"] = False
            out.append(rec)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--thresholds", default="0.90,0.95,0.98,0.99,0.995")
    args = parser.parse_args()

    print(f"{'thr':>6} {'worst FP':>9} {'avg el.rec':>11} {'abstain':>8} {'pass':>5}")
    for thr in (float(t) for t in args.thresholds.split(",")):
        gated = regate(Path(args.preds), thr)
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
            for rec in gated:
                tmp.write(json.dumps(rec) + "\n")
        try:
            res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "eea_query_intent.cli",
                    "evaluate",
                    "--gold",
                    args.gold,
                    "--predictions",
                    tmp.name,
                ],
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
            report = json.loads(res.stdout)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
        langs = report["languages"]
        worst_fp = max(lm["no_ai_false_positive_rate"] for lm in langs.values())
        avg_recall = sum(lm["eligible_recall"] for lm in langs.values()) / len(langs)
        avg_abstain = sum(lm["abstention_rate"] for lm in langs.values()) / len(langs)
        print(
            f"{thr:>6} {worst_fp:>9.3f} {avg_recall:>11.3f} {avg_abstain:>8.3f} "
            f"{str(report['passes']):>5}"
        )


if __name__ == "__main__":
    main()
