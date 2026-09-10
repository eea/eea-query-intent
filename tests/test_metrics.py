import pytest

from eea_query_intent.dataset import DatasetRecord
from eea_query_intent.metrics import (
    AcceptanceThresholds,
    PredictionRecord,
    evaluate_predictions,
)


def gold(
    record_id: str,
    intent: str,
    *,
    language: str = "en",
) -> DatasetRecord:
    return DatasetRecord.from_mapping(
        {
            "id": record_id,
            "language": language,
            "text": f"Query {record_id}",
            "intent": intent,
            "template_id": f"template-{record_id}",
            "source_type": "human_authored",
            "review_status": "native_reviewed",
            "split": "test",
        }
    )


def prediction(
    record_id: str,
    intent: str,
    eligible_probability: float,
    *,
    abstained: bool = False,
) -> PredictionRecord:
    return PredictionRecord.from_mapping(
        {
            "id": record_id,
            "intent": intent,
            "eligible": intent in {"question", "exploratory", "claim"}
            and not abstained,
            "eligible_probability": eligible_probability,
            "confidence": max(eligible_probability, 1 - eligible_probability),
            "abstained": abstained,
            "model_version": "test-model",
            "latency_ms": 10.0,
        }
    )


def test_reports_asymmetric_routing_metrics_and_abstention() -> None:
    records = [
        gold("q", "question"),
        gold("c", "claim"),
        gold("r1", "retrieval"),
        gold("r2", "retrieval"),
    ]
    predictions = [
        prediction("q", "question", 0.9),
        prediction("c", "unknown", 0.4, abstained=True),
        prediction("r1", "exploratory", 0.8),
        prediction("r2", "retrieval", 0.1),
    ]

    report = evaluate_predictions(records, predictions)
    metrics = report["languages"]["en"]

    assert metrics["count"] == 4
    assert metrics["routing_confusion"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    assert metrics["no_ai_false_positive_rate"] == pytest.approx(0.5)
    assert metrics["eligible_precision"] == pytest.approx(0.5)
    assert metrics["eligible_recall"] == pytest.approx(0.5)
    assert metrics["coverage"] == pytest.approx(0.75)
    assert metrics["brier_score"] == pytest.approx(0.255)


def test_reports_each_language_and_worst_language_gate() -> None:
    records = [
        gold("en-q", "question", language="en"),
        gold("en-r", "retrieval", language="en"),
        gold("mt-q", "question", language="mt"),
        gold("mt-r", "retrieval", language="mt"),
    ]
    predictions = [
        prediction("en-q", "question", 0.99),
        prediction("en-r", "retrieval", 0.01),
        prediction("mt-q", "question", 0.99),
        prediction("mt-r", "question", 0.99),
    ]

    report = evaluate_predictions(
        records,
        predictions,
        thresholds=AcceptanceThresholds(
            minimum_eligible_count=1,
            minimum_no_ai_count=1,
            maximum_no_ai_false_positive_rate=0.01,
            minimum_eligible_precision=0.98,
            minimum_macro_f1=0.95,
        ),
    )

    assert report["languages"]["en"]["passes"] is True
    assert report["languages"]["mt"]["passes"] is False
    assert report["passes"] is False
    assert report["failing_languages"] == ["mt"]


def test_rejects_missing_and_extra_predictions() -> None:
    records = [gold("q", "question")]

    with pytest.raises(ValueError, match="missing predictions: q"):
        evaluate_predictions(records, [])

    with pytest.raises(ValueError, match="unknown prediction ids: extra"):
        evaluate_predictions(
            records,
            [prediction("q", "question", 0.9), prediction("extra", "claim", 0.9)],
        )
