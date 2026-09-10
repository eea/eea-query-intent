"""In-sample sanity check: re-predict the TRAIN split and evaluate it.

If in-sample accuracy is high while the held-out template test split is low,
the bottleneck is template generalization (data diversity), not the pipeline.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ELIGIBLE = {"question", "exploratory", "claim"}
LABELS = ("question", "exploratory", "claim", "retrieval", "unknown")


def make_predictor(model_type: str, model_dir: Path):
    if model_type == "setfit":
        from setfit import SetFitModel

        model = SetFitModel.from_pretrained(str(model_dir), device="cpu")

        def predict_setfit(text: str) -> tuple[str, float]:
            probs = model.predict_proba([text], as_numpy=True)[0]
            prob_map = dict(zip(LABELS, (float(p) for p in probs), strict=False))
            top = max(prob_map, key=prob_map.get)
            return top, min(1.0, sum(prob_map[label] for label in ELIGIBLE))

        return predict_setfit

    if model_type == "fasttext":
        import fasttext

        raw = fasttext.load_model(str(model_dir / "model.ftz"))

        def predict_fasttext(text: str) -> tuple[str, float]:
            labels, probs = raw.predict(text.replace("\n", " "))
            prob_map = {label: 0.0 for label in LABELS}
            for label, prob in zip(labels, probs, strict=False):
                prob_map[label] = float(prob)
            top = max(prob_map, key=prob_map.get)
            return top, min(1.0, sum(prob_map[label] for label in ELIGIBLE))

        return predict_fasttext

    raise SystemExit(f"unknown model type: {model_type}")


def main() -> int:
    import sys

    model_type = sys.argv[1] if len(sys.argv) > 1 else "setfit"
    model_dir = ROOT / "models" / model_type
    predict = make_predictor(model_type, model_dir)

    out = model_dir / "train-predictions.jsonl"
    rows = [
        json.loads(line)
        for line in (ROOT / "data/multilingual/v1/train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            started = time.perf_counter()
            intent, eligible_probability = predict(row["text"])
            latency = (time.perf_counter() - started) * 1000
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "intent": intent,
                        "eligible": intent in ELIGIBLE,
                        "eligible_probability": eligible_probability,
                        "confidence": max(
                            eligible_probability,
                            1 - eligible_probability,
                        ),
                        "abstained": False,
                        "model_version": f"{model_type}-v1",
                        "latency_ms": latency,
                    }
                )
                + "\n"
            )
    print(f"wrote {out} ({len(rows)} rows)")

    preds = {
        json.loads(line)["id"]: json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
    }
    conf: Counter = Counter()
    for row in rows:
        if row["language"] != "en":
            continue
        conf[(row["intent"], preds[row["id"]]["intent"])] += 1
    print("\nEN in-sample confusion (gold -> pred):")
    for (gold, pred), count in sorted(conf.items()):
        marker = "ok" if gold == pred else "X "
        print(f"  {marker} {gold:<12s} -> {pred:<12s} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
