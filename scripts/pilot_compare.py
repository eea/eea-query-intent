"""Side-by-side per-language comparison of the current setfit-v4
acceptance report against the NLLB pilot report.

Usage:
    uv run python scripts/pilot_compare.py \
        reports/final_acceptance_report.json \
        reports/pilot_nllb_acceptance_report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PILOT_TARGETS = ("mt", "ga", "is")
COLS = (
    "eligible_recall",
    "eligible_precision",
    "no_ai_false_positive_rate",
    "abstention_rate",
    "macro_f1",
)


def main() -> int:
    cur_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT / "reports" / "final_acceptance_report.json"
    )
    pil_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else ROOT / "reports" / "pilot_nllb_acceptance_report.json"
    )
    current = json.loads(cur_path.read_text())
    pilot = json.loads(pil_path.read_text())

    print(f"current: {cur_path.name} (passes={current.get('passes')})")
    print(f"pilot:   {pil_path.name} (passes={pilot.get('passes')})")
    print()
    header = f"{'lang':4} {'metric':24} {'current':>9} {'pilot':>9} {'delta':>8}"
    print(header)
    for lang in sorted(current["languages"]):
        c = current["languages"][lang]
        p = pilot["languages"].get(lang, {})
        target = "  <-- pilot target" if lang in PILOT_TARGETS else ""
        for col in COLS:
            cv, pv = c.get(col), p.get(col)
            if cv is None or pv is None:
                continue
            delta = pv - cv
            target_mark = target if col == "eligible_recall" else ""
            print(f"{lang:4} {col:24} {cv:9.3f} {pv:9.3f} {delta:+8.3f}{target_mark}")
        print()

    # headline: pilot-target languages
    print("=== HEADLINE (pilot target languages) ===")
    for lang in PILOT_TARGETS:
        c = current["languages"].get(lang, {})
        p = pilot["languages"].get(lang, {})
        print(
            f"{lang}: eligible_recall {c.get('eligible_recall', 0):.3f} -> "
            f"{p.get('eligible_recall', 0):.3f} | "
            f"no_ai_fp {c.get('no_ai_false_positive_rate', 0):.3f} -> "
            f"{p.get('no_ai_false_positive_rate', 0):.3f} | "
            f"abstain {c.get('abstention_rate', 0):.3f} -> "
            f"{p.get('abstention_rate', 0):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
