"""Run the SetFit model over the acceptance test split and write predictions.

Applies the deployed abstain threshold (default from the model manifest) so
the prediction file already carries the gated ``abstained`` / ``eligible``
fields expected by ``eea-query-intent evaluate``.

Usage:
  uv run python scripts/predict_acceptance.py \
      --input data/acceptance/v1/test.jsonl \
      --out models/setfit/acceptance-predictions.jsonl --device mps
"""

import argparse
import json
import time
from pathlib import Path

from eea_query_intent.service import load_adapter

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "data/acceptance/v1/test.jsonl"))
    parser.add_argument("--model", default=str(ROOT / "models/setfit"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="abstain threshold (default: model manifest value)",
    )
    parser.add_argument(
        "--out", default=str(ROOT / "models/setfit/acceptance-predictions.jsonl")
    )
    parser.add_argument(
        "--lowercase-input",
        action="store_true",
        help="lowercase each query before classification (matches the "
        "service-side normalization of the lowercase-trained candidate)",
    )
    args = parser.parse_args()

    adapter, manifest = load_adapter(Path(args.model), args.device)
    threshold = (
        args.threshold
        if args.threshold is not None
        else manifest.get("abstain_threshold", 0.98)
    )
    eligible_labels = set(adapter.eligible_labels)
    model_version = adapter.model_version

    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.lowercase_input:
        for row in rows:
            row["text"] = row["text"].lower()

    # warmup (first call includes lazy graph init)
    adapter.classify("warmup")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for i, row in enumerate(rows, start=1):
            started = time.perf_counter()
            top_intent, eligible_probability = adapter.classify(row["text"])
            latency_ms = (time.perf_counter() - started) * 1000.0
            abstained = eligible_probability < threshold
            if abstained:
                intent = "unknown"
                eligible = False
            else:
                intent = top_intent
                eligible = top_intent in eligible_labels
            record = {
                "id": row["id"],
                "intent": intent,
                "raw_intent": top_intent,
                "eligible": eligible,
                "eligible_probability": eligible_probability,
                "confidence": max(eligible_probability, 1.0 - eligible_probability),
                "abstained": abstained,
                "model_version": model_version,
                "latency_ms": latency_ms,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if i % 1000 == 0:
                print(f"{i}/{len(rows)}", flush=True)

    print(
        f"wrote {out_path} ({len(rows)} predictions, threshold={threshold})", flush=True
    )


if __name__ == "__main__":
    main()
