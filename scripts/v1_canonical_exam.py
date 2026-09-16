"""Run the canonical v2 exam EXACTLY ONCE on the locked finalist.

One inference pass over the whole exam (raw probabilities at threshold
0.0, input lowercased as the production service will lowercase). The
gated view at the selected threshold is re-derived from the same
predictions (sweep_acceptance.regate) — no second inference pass. The
formal per-language report is then computed against the gated view.

Usage:
  uv run python scripts/v1_canonical_exam.py <model-dir> <threshold>
Writes <model-dir>/exam-v2-predictions.jsonl (+ -gated) and
reports/v1_canonical_exam_<model-name>.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from sweep_acceptance import regate  # noqa: E402

EXAM = ROOT / "data" / "acceptance" / "v2" / "test.jsonl"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    model_dir = (ROOT / sys.argv[1]).resolve()
    threshold = float(sys.argv[2])
    name = model_dir.name

    raw_out = model_dir / "exam-v2-predictions.jsonl"
    if raw_out.exists():
        print(f"refusing to run: {raw_out} already exists (exam runs once)")
        return 1

    res = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/predict_acceptance.py",
            "--model",
            str(model_dir),
            "--input",
            str(EXAM),
            "--threshold",
            "0.0",
            "--device",
            "mps",
            "--lowercase-input",
            "--out",
            str(raw_out),
        ],
        cwd=ROOT,
    )
    if res.returncode != 0:
        print("exam inference FAILED")
        return 1

    gated = regate(raw_out, threshold)
    gated_out = model_dir / "exam-v2-predictions-gated.jsonl"
    with gated_out.open("w", encoding="utf-8") as fh:
        for rec in gated:
            fh.write(json.dumps(rec) + "\n")

    report_out = ROOT / "reports" / f"v1_canonical_exam_{name}.json"
    res = subprocess.run(
        [
            "uv",
            "run",
            "eea-query-intent",
            "evaluate",
            "--gold",
            str(EXAM),
            "--predictions",
            str(gated_out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0 or not res.stdout.strip():
        print(f"evaluate FAILED: {res.stderr[:500]}")
        return 1
    report = json.loads(res.stdout)
    report_out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    langs = report["languages"]
    worst = max(langs.values(), key=lambda lm: lm["no_ai_false_positive_rate"])
    avg_recall = sum(lm["eligible_recall"] for lm in langs.values()) / len(langs)
    print(f"exam rows: {sum(lm['count'] for lm in langs.values())}")
    print(f"threshold: {threshold}")
    print(
        f"worst-language no-AI FP: {worst['no_ai_false_positive_rate']:.4f} "
        f"({worst_key(langs, worst)})"
    )
    print(f"avg eligible recall: {avg_recall:.4f}")
    print(f"formal five-class pass (diagnostic only): {report['passes']}")
    print(f"report: {report_out}")
    return 0


def worst_key(langs: dict, worst_lm: dict) -> str:
    for code, lm in langs.items():
        if lm is worst_lm:
            return code
    return "?"


if __name__ == "__main__":
    raise SystemExit(main())
