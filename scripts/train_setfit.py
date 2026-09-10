"""Train the SetFit (multilingual MiniLM L6) candidate and emit predictions.

Usage:
    uv run python scripts/train_setfit.py
    uv run python scripts/train_setfit.py \
        --train-file data/english_only/train.jsonl \
        --calibration-file data/english_only/calibration.jsonl \
        --test-file data/multilingual/v1/test.jsonl \
        --model-dir models/setfit-en --model-version setfit-en-v1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "multilingual" / "v1"
DEFAULT_MODEL_DIR = ROOT / "models" / "setfit"
ELIGIBLE = ("question", "exploratory", "claim")
LABELS = ("question", "exploratory", "claim", "retrieval", "unknown")
BACKBONE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default=str(DEFAULT_DATA / "train.jsonl"))
    parser.add_argument(
        "--calibration-file", default=str(DEFAULT_DATA / "calibration.jsonl")
    )
    parser.add_argument("--test-file", default=str(DEFAULT_DATA / "test.jsonl"))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--model-version", default="setfit-v1")
    return parser.parse_args()


def read_file(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def train(args) -> object:
    import torch
    import torch.nn as nn
    from sentence_transformers import SentenceTransformer
    from setfit import SetFitHead, SetFitModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"training on device: {device}")

    encoder = SentenceTransformer(BACKBONE, device=device)
    # The MPS SDPA kernel does not support dropout; zeroing the p of all
    # dropout modules keeps training on MPS stable. For a frozen-encoder
    # linear head this has a negligible effect on the final classifier.
    for module in encoder.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    head = SetFitHead(in_features=384, out_features=len(LABELS), device=device)
    model = SetFitModel(model_body=encoder, model_head=head, labels=list(LABELS))

    rows = read_file(args.train_file)
    training_data = [row["text"] for row in rows]
    y = [LABELS.index(row["intent"]) for row in rows]
    print(f"training on {len(rows)} rows from {args.train_file}")

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


def predict_row(model, text: str) -> tuple[str, float]:
    probabilities = model.predict_proba([text.replace("\n", " ")], as_numpy=True)[0]
    prob_map = {
        label: float(prob) for label, prob in zip(LABELS, probabilities, strict=False)
    }
    top_intent = max(prob_map, key=prob_map.get)
    eligible_probability = min(
        1.0, max(0.0, sum(prob_map.get(label, 0.0) for label in ELIGIBLE))
    )
    return top_intent, eligible_probability


def emit_predictions(args, split: str, file: str, model) -> Path:
    model_dir = Path(args.model_dir)
    out = model_dir / f"{split}-predictions.jsonl"
    rows = read_file(file)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            started = time.perf_counter()
            intent, eligible_probability = predict_row(model, row["text"])
            latency = (time.perf_counter() - started) * 1000
            prediction = {
                "id": row["id"],
                "intent": intent,
                "eligible": intent in ELIGIBLE,
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
    manifest = {
        "model_type": "setfit",
        "model_version": args.model_version,
        "backbone": BACKBONE,
        "artifact": str(model_dir),
        "labels": list(LABELS),
        "eligible_labels": list(ELIGIBLE),
        "train_file": args.train_file,
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
