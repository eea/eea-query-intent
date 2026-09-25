"""FastAPI service exposing the multilingual query intent classifier.

The service loads the SetFit model through an adapter, applies the
deterministic local policy guards first, then the calibrated abstention
threshold, and returns the shared ``ClassificationResult`` contract. Any
adapter failure is answered with a 503 JSON error so that clients fail
closed (no AI summary) instead of guessing.

Environment:
    EEA_QI_MODEL_PATH        model directory (default: models/setfit)
    EEA_QI_ABSTAIN_THRESHOLD overrides the manifest threshold
    EEA_QI_DEVICE            'cpu' (default), 'mps', or 'cuda'
    EEA_QI_MAX_WORDS         policy guard word limit (default: 20)
    EEA_QI_HOST              bind address (default: 127.0.0.1; containers set 0.0.0.0)
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
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "setfit"


class IntentAdapter(Protocol):
    model_version: str

    def classify(self, text: str) -> tuple[str, float]:
        """Return (intent label, P(eligible)) for one query."""


@dataclass(frozen=True)
class SetFitAdapter:
    model: Any
    model_version: str
    labels: tuple[str, ...]
    eligible_labels: tuple[str, ...]

    def classify(self, text: str) -> tuple[str, float]:
        probabilities = self.model.predict_proba(
            [text.replace("\n", " ")], as_numpy=True
        )[0]
        probabilities = tuple(float(p) for p in probabilities)
        if len(probabilities) != len(self.labels):
            raise ValueError(
                f"model returned {len(probabilities)} probabilities, "
                f"expected {len(self.labels)}"
            )
        prob_map = dict(zip(self.labels, probabilities, strict=True))
        top_intent = max(prob_map, key=prob_map.get)
        eligible_probability = min(
            1.0, max(0.0, sum(prob_map[label] for label in self.eligible_labels))
        )
        return top_intent, eligible_probability


def load_adapter(model_path: Path, device: str):
    """Load the SetFit model and its manifest; returns (adapter, manifest)."""
    manifest = json.loads((model_path / "manifest.json").read_text("utf-8"))
    for key in ("model_version", "labels", "eligible_labels"):
        if key not in manifest:
            raise ValueError(f"manifest missing required key '{key}'")

    from setfit import SetFitModel

    model = SetFitModel.from_pretrained(str(model_path), device=device)
    adapter: IntentAdapter = SetFitAdapter(
        model=model,
        model_version=manifest["model_version"],
        labels=tuple(manifest["labels"]),
        eligible_labels=tuple(manifest["eligible_labels"]),
    )
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

    intent, eligible_probability = adapter.classify(query.casefold())
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
    model_path = Path(os.environ.get("EEA_QI_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    device = os.environ.get("EEA_QI_DEVICE", "cpu")
    max_words = int(os.environ.get("EEA_QI_MAX_WORDS", str(DEFAULT_MAX_WORDS)))

    adapter, manifest = load_adapter(model_path, device)
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
            "model_type": "setfit",
            "model_version": manifest["model_version"],
            "abstain_threshold": threshold,
            "max_words": max_words,
            "uptime_seconds": round(time.time() - started, 1),
        }

    @app.post("/v1/classify", response_model=None)
    def classify(request: ClassifyRequest) -> dict | JSONResponse:
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
    host = os.environ.get("EEA_QI_HOST", "127.0.0.1")
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
