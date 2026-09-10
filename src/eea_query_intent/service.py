"""FastAPI service exposing the multilingual query intent classifier.

The service is model-agnostic: it loads one candidate through an adapter,
applies the deterministic local policy guards first, then the calibrated
abstention threshold, and returns the shared ``ClassificationResult``
contract. Any adapter failure is answered with a 503 JSON error so that
clients fail closed (no AI summary) instead of guessing.

Environment:
    EEA_QI_MODEL_TYPE        'setfit' (default) or 'fasttext'
    EEA_QI_MODEL_PATH        model directory (default: models/<type>)
    EEA_QI_ABSTAIN_THRESHOLD overrides the manifest threshold
    EEA_QI_DEVICE            'cpu' (default), 'mps', or 'cuda'
    EEA_QI_MAX_WORDS         policy guard word limit (default: 20)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from eea_query_intent.contracts import (
    AI_ELIGIBLE_INTENTS,
    ClassificationResult,
    Intent,
)
from eea_query_intent.policy import (
    DEFAULT_MAX_WORDS,
    evaluate_local_policy,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_LABELS = ("question", "exploratory", "claim", "retrieval", "unknown")
ELIGIBLE_LABELS = ("question", "exploratory", "claim")


class IntentAdapter(Protocol):
    model_version: str

    def classify(self, text: str) -> tuple[str, float]:
        """Return (intent label, P(eligible)) for one query."""


@dataclass(frozen=True)
class FastTextAdapter:
    model: Any
    model_version: str

    def classify(self, text: str) -> tuple[str, float]:
        labels, probabilities = self.model.predict(text.replace("\n", " "))
        prob_map = {label: 0.0 for label in MODEL_LABELS}
        for label, probability in zip(labels, probabilities, strict=False):
            prob_map[str(label)] = float(probability)
        top_intent = max(prob_map, key=prob_map.get)
        eligible_probability = min(
            1.0, max(0.0, sum(prob_map[label] for label in ELIGIBLE_LABELS))
        )
        return top_intent, eligible_probability


@dataclass(frozen=True)
class SetFitAdapter:
    model: Any
    model_version: str

    def classify(self, text: str) -> tuple[str, float]:
        probabilities = self.model.predict_proba(
            [text.replace("\n", " ")], as_numpy=True
        )[0]
        prob_map = dict(
            zip(MODEL_LABELS, (float(prob) for prob in probabilities), strict=False)
        )
        top_intent = max(prob_map, key=prob_map.get)
        eligible_probability = min(
            1.0, max(0.0, sum(prob_map[label] for label in ELIGIBLE_LABELS))
        )
        return top_intent, eligible_probability


def load_adapter(model_type: str, model_path: Path, device: str):
    """Load the model and its manifest; returns (adapter, manifest)."""
    manifest = json.loads((model_path / "manifest.json").read_text("utf-8"))

    if model_type == "fasttext":
        import fasttext

        model = fasttext.load_model(str(model_path / "model.ftz"))
        adapter: IntentAdapter = FastTextAdapter(
            model=model, model_version=manifest["model_version"]
        )
    elif model_type == "setfit":
        from setfit import SetFitModel

        model = SetFitModel.from_pretrained(str(model_path), device=device)
        adapter = SetFitAdapter(model=model, model_version=manifest["model_version"])
    else:
        raise ValueError(f"unknown model type: {model_type}")
    return adapter, manifest


def classify_query(
    query: str,
    adapter: IntentAdapter,
    threshold: float,
    *,
    max_words: int = DEFAULT_MAX_WORDS,
) -> ClassificationResult:
    """Run policy guards, the model, and the abstention threshold."""
    model_version = adapter.model_version
    policy = evaluate_local_policy(query, max_words=max_words)
    if not policy.should_classify:
        return ClassificationResult(
            intent=Intent.UNKNOWN,
            eligible=False,
            confidence=1.0,
            eligible_probability=0.0,
            abstained=False,
            reason=policy.reason,
            model_version=model_version,
        )

    intent, eligible_probability = adapter.classify(query)
    eligible_probability = min(1.0, max(0.0, eligible_probability))
    if eligible_probability < threshold:
        return ClassificationResult(
            intent=Intent.UNKNOWN,
            eligible=False,
            confidence=1.0 - eligible_probability,
            eligible_probability=eligible_probability,
            abstained=True,
            reason="below_threshold",
            model_version=model_version,
        )

    intent = Intent(intent)
    eligible = intent in AI_ELIGIBLE_INTENTS
    confidence = eligible_probability if eligible else 1.0 - eligible_probability
    return ClassificationResult(
        intent=intent,
        eligible=eligible,
        confidence=confidence,
        eligible_probability=eligible_probability,
        abstained=False,
        reason="classified",
        model_version=model_version,
    )


class ClassifyRequest(BaseModel):
    query: str = Field(min_length=0, max_length=500)


def create_app() -> FastAPI:
    model_type = os.environ.get("EEA_QI_MODEL_TYPE", "setfit")
    model_path = Path(
        os.environ.get(
            "EEA_QI_MODEL_PATH",
            str(REPO_ROOT / "models" / model_type),
        )
    )
    device = os.environ.get("EEA_QI_DEVICE", "cpu")
    max_words = int(os.environ.get("EEA_QI_MAX_WORDS", str(DEFAULT_MAX_WORDS)))

    adapter, manifest = load_adapter(model_type, model_path, device)
    threshold = float(
        os.environ.get(
            "EEA_QI_ABSTAIN_THRESHOLD",
            str(manifest["abstain_threshold"]),
        )
    )
    started = time.time()

    app = FastAPI(
        title="eea-query-intent",
        version=manifest.get("model_version", "unknown"),
    )
    app.state.adapter = adapter
    app.state.threshold = threshold
    app.state.manifest = manifest
    app.state.max_words = max_words

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_type": model_type,
            "model_version": manifest["model_version"],
            "abstain_threshold": threshold,
            "max_words": max_words,
            "uptime_seconds": round(time.time() - started, 1),
        }

    @app.post("/v1/classify")
    def classify(request: ClassifyRequest) -> dict:
        started_at = time.perf_counter()
        try:
            result = classify_query(
                request.query,
                app.state.adapter,
                app.state.threshold,
                max_words=app.state.max_words,
            )
        except Exception:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "classifier unavailable",
                    "detail": "classification failed; fail closed",
                },
            )
        latency_ms = (time.perf_counter() - started_at) * 1000
        payload = asdict(result)
        payload["intent"] = result.intent.value
        payload["latency_ms"] = round(latency_ms, 2)
        return payload

    return app


def main() -> None:
    import uvicorn

    port = int(os.environ.get("EEA_QI_PORT", "8100"))
    uvicorn.run(create_app(), host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
