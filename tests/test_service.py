"""Tests for the FastAPI service and its classify pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from eea_query_intent.service import classify_query, create_app


class FakeAdapter:
    model_version = "fake-v1"

    def __init__(self, intent: str, probability: float):
        self.intent = intent
        self.probability = probability
        self.calls: list[str] = []

    def classify(self, text: str):
        self.calls.append(text)
        return self.intent, self.probability


class ExplodingAdapter:
    model_version = "boom-v1"

    def classify(self, text: str):
        raise RuntimeError("model exploded")


@pytest.fixture()
def manifest(tmp_path: Path) -> dict:
    manifest = {
        "model_type": "fake",
        "model_version": "fake-v1",
        "abstain_threshold": 0.8,
    }
    (tmp_path / "manifest.json").write_text(
        __import__("json").dumps(manifest), encoding="utf-8"
    )
    return manifest


def test_policy_guard_empty_skips_the_model() -> None:
    adapter = FakeAdapter("question", 0.99)
    result = classify_query("   ", adapter, 0.8)
    assert adapter.calls == []
    assert result.eligible is False
    assert result.reason == "empty"
    assert result.abstained is False


def test_policy_guard_too_long_skips_the_model() -> None:
    adapter = FakeAdapter("question", 0.99)
    query = " ".join(["word"] * 21)
    result = classify_query(query, adapter, 0.8)
    assert adapter.calls == []
    assert result.reason == "too_long"
    assert result.eligible is False


def test_eligible_intent_above_threshold_routes_to_ai() -> None:
    adapter = FakeAdapter("question", 0.95)
    result = classify_query("What is the ozone layer?", adapter, 0.8)
    assert result.eligible is True
    assert result.intent.value == "question"
    assert result.abstained is False
    assert result.reason == "classified"
    assert result.eligible_probability == pytest.approx(0.95)


def test_below_threshold_abstains_and_fails_closed() -> None:
    adapter = FakeAdapter("question", 0.5)
    result = classify_query("What is the ozone layer?", adapter, 0.8)
    assert result.eligible is False
    assert result.abstained is True
    assert result.intent.value == "unknown"
    assert result.reason == "below_threshold"


def test_retrieval_above_threshold_stays_no_ai() -> None:
    adapter = FakeAdapter("retrieval", 0.99)
    result = classify_query("SOER 2025", adapter, 0.8)
    assert result.eligible is False
    assert result.intent.value == "retrieval"
    assert result.abstained is False


def test_eligible_probability_is_clamped() -> None:
    class ClampedAdapter:
        model_version = "clamp-v1"

        def classify(self, text: str):
            return "claim", 1.4

    result = classify_query("A claim.", ClampedAdapter(), 0.8)
    assert result.eligible_probability == 1.0
    assert result.eligible is True


def test_health_and_classify_endpoints(
    monkeypatch, manifest: dict, tmp_path: Path
) -> None:
    from fastapi.testclient import TestClient

    adapter = FakeAdapter("question", 0.99)
    monkeypatch.setenv("EEA_QI_MODEL_TYPE", "fake")
    monkeypatch.setenv("EEA_QI_MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("EEA_QI_ABSTAIN_THRESHOLD", str(manifest["abstain_threshold"]))
    import eea_query_intent.service as service_module

    monkeypatch.setattr(
        service_module,
        "load_adapter",
        lambda model_path, device: (
            adapter,
            manifest,
        ),
    )

    client = TestClient(create_app())
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "fake-v1"
    assert body["abstain_threshold"] == 0.8

    response = client.post("/v1/classify", json={"query": "What is the ozone layer?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["eligible"] is True
    assert payload["intent"] == "question"
    assert payload["latency_ms"] >= 0
    assert adapter.calls == ["What is the ozone layer?"]


def test_policy_guard_reasons_come_from_the_endpoint(
    monkeypatch, manifest: dict
) -> None:
    from fastapi.testclient import TestClient

    adapter = FakeAdapter("question", 0.99)
    import eea_query_intent.service as service_module

    monkeypatch.setattr(
        service_module,
        "load_adapter",
        lambda model_path, device: (adapter, manifest),
    )
    client = TestClient(create_app())

    empty = client.post("/v1/classify", json={"query": ""})
    assert empty.status_code == 200
    assert empty.json()["reason"] == "empty"
    assert adapter.calls == []


def test_adapter_error_returns_503_fail_closed(monkeypatch, manifest: dict) -> None:
    from fastapi.testclient import TestClient

    import eea_query_intent.service as service_module

    monkeypatch.setattr(
        service_module,
        "load_adapter",
        lambda model_path, device: (
            ExplodingAdapter(),
            manifest,
        ),
    )
    client = TestClient(create_app())
    response = client.post("/v1/classify", json={"query": "What is the ozone layer?"})
    assert response.status_code == 503
    assert response.json()["error"] == "classifier unavailable"
