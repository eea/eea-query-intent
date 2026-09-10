"""Regenerate test predictions with the calibrated abstention threshold.

Reads each model's raw test predictions and manifest threshold, marks
rows below the threshold as abstained (not eligible, intent unknown),
and writes <split>-predictions-gated.jsonl for the evaluate CLI.

Usage:
    uv run python scripts/apply_threshold.py fasttext
    uv run python scripts/apply_threshold.py setfit
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    import sys

    if len(sys.argv) != 2 or sys.argv[1] not in {"fasttext", "setfit"}:
        raise SystemExit(
            "usage: uv run python scripts/apply_threshold.py <fasttext|setfit>"
        )
    model_type = sys.argv[1]
    model_dir = ROOT / "models" / model_type
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    threshold = float(manifest["abstain_threshold"])

    for split in ("test", "calibration"):
        src = model_dir / f"{split}-predictions.jsonl"
        dst = model_dir / f"{split}-predictions-gated.jsonl"
        count_abstained = 0
        with (
            src.open(encoding="utf-8") as handle,
            dst.open("w", encoding="utf-8") as out,
        ):
            for line in handle:
                if not line.strip():
                    continue
                prediction = json.loads(line)
                probability = prediction["eligible_probability"]
                if probability < threshold:
                    count_abstained += 1
                    prediction["abstained"] = True
                    prediction["eligible"] = False
                    prediction["intent"] = "unknown"
                    prediction["confidence"] = 1.0 - probability
                else:
                    prediction["confidence"] = probability
                out.write(json.dumps(prediction) + "\n")
        print(
            f"{model_type}/{split}: threshold={threshold} abstained={count_abstained}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
