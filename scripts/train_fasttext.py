"""Train the fastText candidate and emit contract-format predictions.

Usage:
    uv run python scripts/train_fasttext.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import fasttext

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "multilingual" / "v1"
MODEL_DIR = ROOT / "models" / "fasttext"
ELIGIBLE = ("question", "exploratory", "claim")
LABELS = ("question", "exploratory", "claim", "retrieval", "unknown")


def read_split(split: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def train() -> fasttext.FastText:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_file = MODEL_DIR / "train.txt"
    with train_file.open("w", encoding="utf-8") as handle:
        for row in read_split("train"):
            handle.write(f"{row['text']} __label__{row['intent']}\n")

    model = fasttext.train_supervised(
        input=str(train_file),
        dim=100,
        wordNgrams=2,
        minCount=1,
        epoch=25,
        lr=0.5,
        loss="softmax",
        thread=8,
        verbose=1,
    )

    quantized = MODEL_DIR / "model.ftz"
    model.quantize(input=str(train_file), retrain=True, qnorm=True, cutoff=100_000)
    model.save_model(str(quantized))
    return model


def predict_row(model: fasttext.FastText, text: str) -> tuple[str, float]:
    labels, probabilities = model.predict(text.replace("\n", " "), k=5)
    label_map = {
        label.removeprefix("__label__"): float(probability)
        for label, probability in zip(labels, probabilities, strict=False)
    }
    top_intent = max(label_map, key=label_map.get) if label_map else "unknown"
    eligible_probability = min(
        1.0, max(0.0, sum(label_map.get(label, 0.0) for label in ELIGIBLE))
    )
    return top_intent, eligible_probability


def emit_predictions(split: str, model: fasttext.FastText) -> Path:
    out = MODEL_DIR / f"{split}-predictions.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in read_split(split):
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
                "model_version": "fasttext-v1",
                "latency_ms": latency,
            }
            handle.write(json.dumps(prediction) + "\n")
    return out


def write_manifest(model: fasttext.FastText) -> None:
    manifest = {
        "model_type": "fasttext",
        "model_version": "fasttext-v1",
        "artifact": str(MODEL_DIR / "model.ftz"),
        "labels": list(LABELS),
        "eligible_labels": list(ELIGIBLE),
    }
    (MODEL_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> int:
    model = train()
    write_manifest(model)
    for split in ("test", "calibration"):
        out = emit_predictions(split, model)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
