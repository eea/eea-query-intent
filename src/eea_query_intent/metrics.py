"""Per-language asymmetric evaluation of intent routing predictions.

The safety-critical decision is binary (AI-eligible vs no-AI), so the primary
metrics are the no-AI false-positive rate and eligible precision. The fine
intents are reported separately for diagnostics. A benchmark only passes when
every language passes, because the deployment requirement is per-language.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from eea_query_intent.contracts import AI_ELIGIBLE_INTENTS, Intent
from eea_query_intent.dataset import DatasetRecord

ELIGIBLE_INTENTS = frozenset(intent.value for intent in AI_ELIGIBLE_INTENTS)


class PredictionValidationError(ValueError):
    """Raised when a prediction row is malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    id: str
    intent: str
    eligible: bool
    eligible_probability: float
    confidence: float
    abstained: bool
    model_version: str
    latency_ms: float

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> PredictionRecord:
        required = (
            "id",
            "intent",
            "eligible",
            "eligible_probability",
            "confidence",
            "abstained",
            "model_version",
            "latency_ms",
        )
        for field in required:
            if field not in row:
                raise PredictionValidationError(f"missing prediction field '{field}'")

        record_id = row["id"]
        intent = row["intent"]
        eligible = row["eligible"]
        probability = row["eligible_probability"]
        confidence = row["confidence"]
        abstained = row["abstained"]

        if not isinstance(record_id, str) or not record_id.strip():
            raise PredictionValidationError("'id' must be a non-empty string")
        if intent not in {member.value for member in Intent}:
            raise PredictionValidationError(f"unsupported intent '{intent}'")
        if not isinstance(eligible, bool):
            raise PredictionValidationError("'eligible' must be a boolean")
        if not isinstance(abstained, bool):
            raise PredictionValidationError("'abstained' must be a boolean")
        for field_name, value in (
            ("eligible_probability", probability),
            ("confidence", confidence),
            ("latency_ms", row["latency_ms"]),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
            ):
                raise PredictionValidationError(
                    f"'{field_name}' must be a non-negative number"
                )
        if not 0.0 <= probability <= 1.0:
            raise PredictionValidationError(
                "'eligible_probability' must be between 0 and 1"
            )

        expected_eligible = intent in ELIGIBLE_INTENTS and not abstained
        if eligible != expected_eligible:
            raise PredictionValidationError(
                f"prediction '{record_id}': 'eligible' must agree with "
                f"intent '{intent}' and abstention"
            )

        return cls(
            id=record_id,
            intent=intent,
            eligible=eligible,
            eligible_probability=probability,
            confidence=confidence,
            abstained=abstained,
            model_version=str(row["model_version"]),
            latency_ms=float(row["latency_ms"]),
        )


@dataclass(frozen=True, slots=True)
class AcceptanceThresholds:
    minimum_eligible_count: int = 300
    minimum_no_ai_count: int = 300
    maximum_no_ai_false_positive_rate: float = 0.01
    minimum_eligible_precision: float = 0.98
    minimum_macro_f1: float = 0.95


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def _evaluate_language(
    records: list[DatasetRecord],
    predictions: list[PredictionRecord],
    thresholds: AcceptanceThresholds,
) -> dict[str, Any]:
    by_id = {prediction.id: prediction for prediction in predictions}

    tp = tn = fp = fn = 0
    brier_sum = 0.0
    abstained = 0
    latencies: list[float] = []
    class_counts: dict[str, tuple[int, int, int]] = {}
    intent_tp: dict[str, int] = {}
    intent_gold: dict[str, int] = {}

    for record in records:
        prediction = by_id[record.id]
        gold_eligible = record.eligible
        predicted_eligible = prediction.eligible

        if gold_eligible and predicted_eligible:
            tp += 1
        elif gold_eligible and not predicted_eligible:
            fn += 1
        elif not gold_eligible and predicted_eligible:
            fp += 1
        else:
            tn += 1

        brier_sum += (prediction.eligible_probability - int(gold_eligible)) ** 2
        abstained += int(prediction.abstained)
        latencies.append(prediction.latency_ms)

        if record.intent == prediction.intent:
            tp_c, fp_c, fn_c = class_counts.get(record.intent, (0, 0, 0))
            class_counts[record.intent] = (tp_c + 1, fp_c, fn_c)
        else:
            tp_c, fp_c, fn_c = class_counts.get(record.intent, (0, 0, 0))
            class_counts[record.intent] = (tp_c, fp_c, fn_c + 1)
            tp_c, fp_c, fn_c = class_counts.get(prediction.intent, (0, 0, 0))
            class_counts[prediction.intent] = (tp_c, fp_c + 1, fn_c)

        if record.eligible:
            intent_gold[record.intent] = intent_gold.get(record.intent, 0) + 1
            if record.intent == prediction.intent and prediction.eligible:
                intent_tp[record.intent] = intent_tp.get(record.intent, 0) + 1

    no_ai_total = fp + tn
    eligible_predicted = tp + fp
    eligible_gold = tp + fn

    false_positive_rate = fp / no_ai_total if no_ai_total else 0.0
    precision = tp / eligible_predicted if eligible_predicted else 1.0
    recall = tp / eligible_gold if eligible_gold else 1.0

    macro_f1 = (
        sum(_f1(tp_c, fp_c, fn_c) for tp_c, fp_c, fn_c in class_counts.values())
        / len(class_counts)
        if class_counts
        else 0.0
    )

    intent_recall = {
        intent: (
            intent_tp.get(intent, 0) / intent_gold[intent]
            if intent in intent_gold
            else None
        )
        for intent in ("question", "exploratory", "claim")
    }

    eligible_count = sum(1 for record in records if record.eligible)
    no_ai_count = len(records) - eligible_count

    failures: list[str] = []
    if eligible_count < thresholds.minimum_eligible_count:
        failures.append("minimum_eligible_count")
    if no_ai_count < thresholds.minimum_no_ai_count:
        failures.append("minimum_no_ai_count")
    if false_positive_rate > thresholds.maximum_no_ai_false_positive_rate:
        failures.append("maximum_no_ai_false_positive_rate")
    if precision < thresholds.minimum_eligible_precision:
        failures.append("minimum_eligible_precision")
    if macro_f1 < thresholds.minimum_macro_f1:
        failures.append("minimum_macro_f1")

    sorted_latencies = sorted(latencies)
    return {
        "count": len(records),
        "eligible_count": eligible_count,
        "no_ai_count": no_ai_count,
        "routing_confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "no_ai_false_positive_rate": false_positive_rate,
        "eligible_precision": precision,
        "eligible_recall": recall,
        "intent_recall": intent_recall,
        "macro_f1": macro_f1,
        "abstention_rate": abstained / len(records) if records else 0.0,
        "coverage": 1 - (abstained / len(records) if records else 0.0),
        "brier_score": brier_sum / len(records) if records else 0.0,
        "latency_p50_ms": _percentile(sorted_latencies, 0.5),
        "latency_p95_ms": _percentile(sorted_latencies, 0.95),
        "failures": failures,
        "passes": not failures,
    }


def evaluate_predictions(
    gold: list[DatasetRecord],
    predictions: list[PredictionRecord],
    thresholds: AcceptanceThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or AcceptanceThresholds()

    gold_ids = {record.id for record in gold}
    prediction_ids = {prediction.id for prediction in predictions}

    missing = sorted(gold_ids - prediction_ids)
    if missing:
        raise ValueError(f"missing predictions: {', '.join(missing)}")
    unknown = sorted(prediction_ids - gold_ids)
    if unknown:
        raise ValueError(f"unknown prediction ids: {', '.join(unknown)}")

    gold_language = {record.id: record.language for record in gold}

    languages: dict[str, Any] = {}
    for language in sorted({language for language in gold_language.values()}):
        languages[language] = _evaluate_language(
            [record for record in gold if record.language == language],
            [
                prediction
                for prediction in predictions
                if gold_language[prediction.id] == language
            ],
            thresholds,
        )

    failing = [
        language for language, metrics in languages.items() if not metrics["passes"]
    ]

    return {
        "languages": languages,
        "passes": not failing,
        "failing_languages": failing,
    }
