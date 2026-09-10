"""Train the SetFit (multilingual MiniLM L6) candidate and emit predictions.

Usage:
    uv run python scripts/train_setfit.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "multilingual" / "v1"
MODEL_DIR = ROOT / "models" / "setfit"
ELIGIBLE = ("question", "exploratory", "claim")
LABELS = ("question", "exploratory", "claim", "retrieval", "unknown")
BACKBONE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def read_split(split: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def train():
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

    rows = read_split("train")
    training_data = [row["text"] for row in rows]
    y = [LABELS.index(row["intent"]) for row in rows]

    model.fit(
        training_data,
        y,
        num_epochs=2,
        batch_size=64,
        head_learning_rate=1e-2,
        body_learning_rate=1e-5,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_DIR))
    return model


def load_model():
    import torch
    from setfit import SetFitModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SetFitModel.from_pretrained(str(MODEL_DIR), device=device)


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


def emit_predictions(split: str, model) -> Path:
    out = MODEL_DIR / f"{split}-predictions.jsonl"
    rows = read_split(split)
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
                "model_version": "setfit-v1",
                "latency_ms": latency,
            }
            handle.write(json.dumps(prediction) + "\n")
    print(f"wrote {out}")
    return out


def write_manifest() -> None:
    manifest = {
        "model_type": "setfit",
        "model_version": "setfit-v1",
        "backbone": BACKBONE,
        "artifact": str(MODEL_DIR),
        "labels": list(LABELS),
        "eligible_labels": list(ELIGIBLE),
    }
    (MODEL_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> int:
    model = train()
    write_manifest()
    emit_predictions("test", model)
    emit_predictions("calibration", model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
