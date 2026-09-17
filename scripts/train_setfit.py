"""Train a SetFit candidate and emit predictions.

Usage (production v1 mix):
    uv run python scripts/train_setfit.py \
        --train-file data/pilot/v1/train.jsonl \
        --calibration-file data/pilot/v1/calibration.jsonl \
        --model-dir models/setfit-v1e5-s3 --model-version setfit-v1 \
        --backbone intfloat/multilingual-e5-small --seed 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "pilot" / "v1"
DEFAULT_MODEL_DIR = ROOT / "models" / "setfit"
ELIGIBLE = ("question", "exploratory", "claim")
LABELS = ("question", "exploratory", "claim", "retrieval", "unknown")
BINARY_LABELS = ("eligible", "ineligible")
BACKBONE = "intfloat/multilingual-e5-small"  # production backbone (setfit-v1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default=str(DEFAULT_DATA / "train.jsonl"))
    parser.add_argument(
        "--calibration-file", default=str(DEFAULT_DATA / "calibration.jsonl")
    )
    parser.add_argument("--test-file", default=str(DEFAULT_DATA / "test.jsonl"))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--model-version", default="setfit-v1")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--backbone", default=BACKBONE)
    parser.add_argument(
        "--binary",
        action="store_true",
        help="objective-aligned binary head (eligible vs ineligible)",
    )
    return parser.parse_args()


def read_file(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def train(args) -> object:
    import random

    import numpy as np
    import torch
    import torch.nn as nn
    from sentence_transformers import SentenceTransformer
    from setfit import SetFitHead, SetFitModel

    if args.seed is not None:
        # Fixed-seed discipline (pre-registration): head init and data
        # shuffling are the stochastic parts of SetFit training.
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"training on device: {device}")

    encoder = SentenceTransformer(args.backbone, device=device)
    # The MPS SDPA kernel does not support dropout; zeroing the p of all
    # dropout modules keeps training on MPS stable. For a frozen-encoder
    # linear head this has a negligible effect on the final classifier.
    for module in encoder.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    labels = list(BINARY_LABELS) if args.binary else list(LABELS)
    dim = encoder.get_sentence_embedding_dimension()
    head = SetFitHead(in_features=dim, out_features=len(labels), device=device)
    model = SetFitModel(model_body=encoder, model_head=head, labels=labels)

    rows = read_file(args.train_file)
    training_data = [row["text"] for row in rows]
    if args.binary:
        y = [0 if row["intent"] in ELIGIBLE else 1 for row in rows]
    else:
        y = [LABELS.index(row["intent"]) for row in rows]
    print(
        f"training on {len(rows)} rows from {args.train_file} "
        f"(binary={args.binary}, backbone={args.backbone}, seed={args.seed})"
    )

    model.fit(
        training_data,
        y,
        num_epochs=2,
        batch_size=64,
        head_learning_rate=1e-2,
        body_learning_rate=1e-5,
    )
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_dir))
    return model


def load_model(args):
    import torch
    from setfit import SetFitModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SetFitModel.from_pretrained(str(Path(args.model_dir)), device=device)


def predict_row(model, text: str, labels: tuple[str, ...]) -> tuple[str, float]:
    probabilities = model.predict_proba([text.replace("\n", " ")], as_numpy=True)[0]
    prob_map = {
        label: float(prob) for label, prob in zip(labels, probabilities, strict=False)
    }
    top_intent = max(prob_map, key=prob_map.get)
    if len(labels) == 2:  # binary head: P(eligible) is the eligible probability
        eligible_probability = min(1.0, max(0.0, prob_map.get("eligible", 0.0)))
    else:
        eligible_probability = min(
            1.0, max(0.0, sum(prob_map.get(label, 0.0) for label in ELIGIBLE))
        )
    return top_intent, eligible_probability


def emit_predictions(args, split: str, file: str, model) -> Path:
    labels = BINARY_LABELS if args.binary else LABELS
    is_binary = len(labels) == 2
    model_dir = Path(args.model_dir)
    out = model_dir / f"{split}-predictions.jsonl"
    rows = read_file(file)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            started = time.perf_counter()
            intent, eligible_probability = predict_row(model, row["text"], labels)
            latency = (time.perf_counter() - started) * 1000
            # binary head: the argmax label IS the routing decision
            eligible = intent == "eligible" if is_binary else intent in ELIGIBLE
            prediction = {
                "id": row["id"],
                "intent": intent,
                "eligible": eligible,
                "eligible_probability": eligible_probability,
                "confidence": max(eligible_probability, 1 - eligible_probability),
                "abstained": False,
                "model_version": args.model_version,
                "latency_ms": latency,
            }
            handle.write(json.dumps(prediction) + "\n")
    print(f"wrote {out} ({len(rows)} rows)")
    return out


def write_manifest(args) -> None:
    model_dir = Path(args.model_dir)
    labels = list(BINARY_LABELS) if args.binary else list(LABELS)
    eligible_labels = ["eligible"] if args.binary else list(ELIGIBLE)
    manifest = {
        "model_type": "setfit-binary" if args.binary else "setfit",
        "model_version": args.model_version,
        "backbone": args.backbone,
        "artifact": str(model_dir),
        "labels": labels,
        "eligible_labels": eligible_labels,
        "train_file": args.train_file,
        "seed": args.seed,
    }
    (model_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    model = train(args)
    write_manifest(args)
    emit_predictions(args, "test", args.test_file, model)
    emit_predictions(args, "calibration", args.calibration_file, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
