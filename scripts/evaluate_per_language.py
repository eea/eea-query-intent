"""Per-language gated evaluation of an English-trained model.

Joins a held-out multilingual gold split with a model's predictions,
applies an abstain threshold (rows with eligible_probability below the
threshold are treated as no-AI / abstained), and reports, per language:

  - n                     total rows
  - no_ai / fp / fp_rate  no-AI rows, no-AI rows falsely routed to AI,
                          and that rate (the safety gate; must be <= 1%)
  - elig / hit / recall   eligible rows, eligible rows routed to AI,
                          and that recall (the abstention cost)
  - abstained             total rows below threshold
  - PASS if fp_rate <= 1%

Usage:
    uv run python scripts/evaluate_per_language.py \
        --gold data/acceptance/v2/test.jsonl \
        --predictions models/setfit/exam-v2-predictions-gated.jsonl \
        --threshold 0.95 [--report reports/<name>.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--threshold", type=float, default=0.86)
    parser.add_argument("--max-no-ai-false-positive-rate", type=float, default=0.01)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    gold = load(args.gold)
    preds = {row["id"]: row for row in load(args.predictions)}

    per: dict[str, dict] = defaultdict(
        lambda: {
            "n": 0,
            "no_ai": 0,
            "fp": 0,
            "elig": 0,
            "hit": 0,
            "abstained": 0,
        }
    )
    for row in gold:
        pred = preds[row["id"]]
        lang = row["language"]
        gold_eligible = row["intent"] in ("question", "exploratory", "claim")
        routed_ai = pred["eligible_probability"] >= args.threshold
        bucket = per[lang]
        bucket["n"] += 1
        if not routed_ai:
            bucket["abstained"] += 1
        if gold_eligible:
            bucket["elig"] += 1
            if routed_ai:
                bucket["hit"] += 1
        else:
            bucket["no_ai"] += 1
            if routed_ai:
                bucket["fp"] += 1

    table = []
    for lang, b in per.items():
        fp_rate = b["fp"] / b["no_ai"] if b["no_ai"] else 0.0
        recall = b["hit"] / b["elig"] if b["elig"] else 0.0
        table.append(
            {
                "language": lang,
                "n": b["n"],
                "no_ai": b["no_ai"],
                "no_ai_false_positives": b["fp"],
                "no_ai_fp_rate": round(fp_rate, 4),
                "eligible": b["elig"],
                "eligible_hit": b["hit"],
                "eligible_recall": round(recall, 4),
                "abstained": b["abstained"],
                "pass": fp_rate <= args.max_no_ai_false_positive_rate,
            }
        )
    table.sort(key=lambda r: (-r["no_ai_false_positives"], r["language"]))

    passed = [r for r in table if r["pass"]]
    failed = [r for r in table if not r["pass"]]
    print(
        f"threshold={args.threshold}   gate: no-AI FP rate <= "
        f"{args.max_no_ai_false_positive_rate:.0%}\n"
    )
    header = (
        f"{'lang':5s} {'n':>3s} {'noAI':>4s} {'FP':>3s} {'FP%':>6s} "
        f"{'elig':>4s} {'rec':>5s} {'abst':>4s}  status"
    )
    print(header)
    print("-" * len(header))
    for r in table:
        flag = "PASS" if r["pass"] else "FAIL"
        print(
            f"{r['language']:5s} {r['n']:3d} {r['no_ai']:4d} "
            f"{r['no_ai_false_positives']:3d} "
            f"{r['no_ai_fp_rate'] * 100:5.1f}% {r['eligible']:4d} "
            f"{r['eligible_recall'] * 100:4.0f}% {r['abstained']:4d}  {flag}"
        )
    print("-" * len(header))
    print(
        f"PASS {len(passed)}/{len(table)} languages | "
        f"FAIL: {', '.join(r['language'] for r in failed) or 'none'}"
    )

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "threshold": args.threshold,
                    "max_no_ai_false_positive_rate": args.max_no_ai_false_positive_rate,
                    "gold": args.gold,
                    "predictions": args.predictions,
                    "passed": len(passed),
                    "total": len(table),
                    "failing_languages": [r["language"] for r in failed],
                    "table": table,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
