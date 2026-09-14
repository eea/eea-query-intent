"""Apply the chosen abstention threshold to the setfit model manifest.

The service reads ``abstain_threshold`` from ``models/setfit/manifest.json``
(``EEA_QI_ABSTAIN_THRESHOLD`` overrides it). Also re-gates the legacy
test/calibration prediction files when present, for continuity with the
old regression split.

Usage: uv run python scripts/apply_threshold.py --threshold 0.99
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "setfit"


def gate_split(src: Path, dst: Path, threshold: float) -> None:
    count = 0
    with src.open(encoding="utf-8") as handle, dst.open("w", encoding="utf-8") as out:
        for line in handle:
            if not line.strip():
                continue
            prediction = json.loads(line)
            probability = prediction["eligible_probability"]
            if probability < threshold:
                count += 1
                prediction["abstained"] = True
                prediction["eligible"] = False
                prediction["intent"] = "unknown"
            prediction["confidence"] = max(probability, 1.0 - probability)
            out.write(json.dumps(prediction) + "\n")
    print(f"setfit/{dst.stem}: threshold={threshold} abstained={count}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()

    manifest_path = MODEL_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["abstain_threshold"] = args.threshold
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"setfit: abstain_threshold={args.threshold} written to manifest")

    for split in ("test", "calibration"):
        src = MODEL_DIR / f"{split}-predictions.jsonl"
        if not src.exists():
            continue
        dst = MODEL_DIR / f"{split}-predictions-gated.jsonl"
        gate_split(src, dst, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
