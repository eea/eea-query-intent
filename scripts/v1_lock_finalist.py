"""Lock the iteration-v1 finalist across families (calibration only).

Pre-registered rule: per family, v1_select_seed.py already picked the
seed (smallest calibration threshold meeting the worst-language 1%
no-AI false-positive gate, then highest calibration eligible recall).
This script ranks the per-family selections with the same rule and
locks exactly one finalist. The exam is never read here.

Usage: uv run python scripts/v1_lock_finalist.py
Writes models/finalist/lock.json and prints:
  LOCKED=<model-dir>
  THRESHOLD=<threshold>
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAMILIES = {
    "d1": ROOT / "models" / "setfit-v1" / "selection.json",
    "binary": ROOT / "models" / "setfit-v1b" / "selection.json",
    "backbone": ROOT / "models" / "setfit-v1e5" / "selection.json",
}


def main() -> int:
    entries: list[dict] = []
    for family, path in FAMILIES.items():
        if not path.exists():
            print(f"skip {family}: no {path}")
            continue
        sel = json.loads(path.read_text(encoding="utf-8"))
        seed = sel["selected"]
        row = next(
            (s for s in sel["seeds"] if s["seed"] == seed),
            None,
        )
        if row is None:
            print(f"skip {family}: selected seed missing from seeds")
            continue
        entries.append(
            {
                "family": family,
                "seed": seed,
                "threshold": sel["selected_threshold"],
                "calib_worst_fp": row["calib_worst_fp"],
                "calib_avg_recall": row["calib_avg_recall"],
                "meets_fp_gate": row["meets_fp_gate"],
                "all_seeds_qualify": sel["all_qualify"],
            }
        )
        print(
            f"{family}: {seed} thr={sel['selected_threshold']} "
            f"worst_fp={row['calib_worst_fp']} recall={row['calib_avg_recall']}"
        )
    if not entries:
        print("no family selections found")
        return 1
    qualifying = [e for e in entries if e["meets_fp_gate"]]
    pool = qualifying or entries
    best = max(pool, key=lambda e: e["calib_avg_recall"])
    lock = {
        "rule": (
            "per-family seed rule applied across families: qualify on the "
            "calibration worst-language 1% no-AI FP gate, then rank by "
            "calibration eligible recall (never by lowest FP)"
        ),
        "qualifying_families": len(qualifying),
        "finalist": best["seed"],
        "family": best["family"],
        "threshold": best["threshold"],
        "entries": entries,
    }
    out = ROOT / "models" / "finalist" / "lock.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"LOCKED={best['seed']}")
    print(f"THRESHOLD={best['threshold']}")
    print(f"lock: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
