"""Select the iteration-v1 seed on the CALIBRATION set only.

Per the pre-registration: the deployment threshold is re-derived per seed
from calibration (choose_final_threshold.py, smallest threshold meeting
the worst-language 1% no-AI false-positive gate), and qualifying seeds
are ranked by calibration eligible recall (never by lowest FP, which
rewards unnecessary abstention). The exam is never used here.

Usage: uv run python scripts/v1_select_seed.py [d1|binary|backbone]
Writes models/<family-prefix>/selection.json and prints SELECTED=<dir>.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from choose_final_threshold import measure  # noqa: E402

PREFIXES = {
    "d1": "setfit-v1",
    "binary": "setfit-v1b",
    "backbone": "setfit-v1e5",
}
CALIB = ROOT / "data" / "pilot" / "v1" / "calibration.jsonl"


def family_dirs(family: str) -> tuple[str, list[str]]:
    prefix = PREFIXES[family]
    seeds = [f"models/{prefix}-s{i}" for i in (1, 2, 3)]
    return prefix, seeds


def main() -> int:
    family = sys.argv[1] if len(sys.argv) > 1 else "d1"
    prefix, seeds = family_dirs(family)
    out = ROOT / "models" / prefix / "selection.json"
    rows: list[dict] = []
    for seed in seeds:
        tpath = ROOT / seed / "threshold.json"
        ppath = ROOT / seed / "calibration-predictions.jsonl"
        if not (tpath.exists() and ppath.exists()):
            print(f"skip {seed}: missing threshold or calibration preds")
            continue
        thr = json.loads(tpath.read_text(encoding="utf-8"))["threshold"]
        m = measure(CALIB, ppath, thr)
        rows.append(
            {
                "family": family,
                "seed": seed,
                "threshold": thr,
                "calib_worst_fp": round(m["worst_fp"], 4),
                "calib_avg_recall": round(m["avg_recall"], 4),
                "calib_avg_abstain": round(m["avg_abstain"], 4),
                "meets_fp_gate": m["worst_fp"] <= 0.01,
            }
        )
        print(
            f"{seed}: thr={thr} worst_fp={m['worst_fp']:.4f} "
            f"recall={m['avg_recall']:.4f} abstain={m['avg_abstain']:.4f}"
        )
    if not rows:
        print("no seeds to select")
        return 1
    qualifying = [r for r in rows if r["meets_fp_gate"]]
    pool = qualifying or rows
    best = max(pool, key=lambda r: r["calib_avg_recall"])
    report = {
        "family": family,
        "rule": (
            "smallest calibration threshold meeting the worst-language 1% "
            "no-AI FP gate; qualify, then rank by calibration eligible recall"
        ),
        "seeds": rows,
        "all_qualify": bool(qualifying),
        "selected": best["seed"],
        "selected_threshold": best["threshold"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SELECTED={best['seed']} threshold={best['threshold']}")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
