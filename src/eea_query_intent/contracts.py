from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Intent(StrEnum):
    QUESTION = "question"
    EXPLORATORY = "exploratory"
    CLAIM = "claim"
    RETRIEVAL = "retrieval"
    UNKNOWN = "unknown"


AI_ELIGIBLE_INTENTS = frozenset({Intent.QUESTION, Intent.EXPLORATORY, Intent.CLAIM})
MODEL_INTENTS = frozenset({*AI_ELIGIBLE_INTENTS, Intent.RETRIEVAL})


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    intent: Intent
    eligible: bool
    confidence: float
    eligible_probability: float
    abstained: bool
    reason: str
    model_version: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("confidence", self.confidence),
            ("eligible_probability", self.eligible_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")

        if self.abstained and (self.eligible or self.intent is not Intent.UNKNOWN):
            raise ValueError("abstention must fail closed with unknown intent")

        expected_eligible = self.intent in AI_ELIGIBLE_INTENTS
        if not self.abstained and self.eligible != expected_eligible:
            raise ValueError("eligible must agree with the predicted intent")
