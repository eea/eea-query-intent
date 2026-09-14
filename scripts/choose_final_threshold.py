"""Select the deployment threshold for the final model.

Re-gates the raw acceptance predictions at each candidate threshold and
picks the SMALLEST threshold whose worst-language no-AI false-positive
rate is at most 1% (the routing safety gate). If no candidate meets the
gate, falls back to 0.98 and flags it in the output.

Usage:
  uv run python scripts/choose_final_threshold.py \
    --gold data/acceptance/v1/test.jsonl \
    --preds models/setfit/acceptance-predictions.jsonl

Writes reports/final_threshold.json and prints CHOSEN=<threshold>.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from sweep_acceptance import regate  # noqa: E402

CANDIDATES = [
    0.80,
    0.82,
    0.84,
    0.85,
    0.86,
    0.88,
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
]
MAX_WORST_FP = 0.01
FALLBACK = 0.98


def measure(gold: Path, preds: Path, thr: float) -> dict:
    gated = regate(preds, thr)
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
                str(gold),
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
    return {
        "worst_fp": max(lm["no_ai_false_positive_rate"] for lm in langs.values()),
        "avg_recall": sum(lm["eligible_recall"] for lm in langs.values()) / len(langs),
        "avg_abstain": sum(lm["abstention_rate"] for lm in langs.values()) / len(langs),
        "formal_pass": report["passes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--preds", required=True)
    args = parser.parse_args()
    gold = Path(args.gold)
    preds = Path(args.preds)

    rows = []
    for thr in CANDIDATES:
        m = measure(gold, preds, thr)
        m["threshold"] = thr
        m["meets_fp_gate"] = m["worst_fp"] <= MAX_WORST_FP
        rows.append(m)
        print(
            f"{thr:>6} worst_fp={m['worst_fp']:.3f} "
            f"recall={m['avg_recall']:.3f} abstain={m['avg_abstain']:.3f} "
            f"gate={'yes' if m['meets_fp_gate'] else 'no'}",
            flush=True,
        )

    meeting = [r for r in rows if r["meets_fp_gate"]]
    if meeting:
        chosen = min(r["threshold"] for r in meeting)
        fallback = False
    else:
        chosen = FALLBACK
        fallback = True

    out = {
        "threshold": chosen,
        "fallback": fallback,
        "gate": "worst-language no-AI false-positive rate <= 1%",
        "candidates": rows,
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    with (reports / "final_threshold.json").open("w", encoding="utf-8") as h:
        json.dump(out, h, indent=2)
        h.write("\n")
    print(f"CHOSEN={chosen} fallback={fallback}")


if __name__ == "__main__":
    main()
