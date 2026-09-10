"""Versioned JSONL dataset format for intent benchmarking.

Each record carries a stable ``template_id`` so translated or generated rows
derived from the same semantic template can be grouped. The split rules for
template groups are the primary defence against translation leakage inflating
per-language accuracy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eea_query_intent.languages import SUPPORTED_LANGUAGE_CODES

ALLOWED_INTENTS = frozenset(
    {"question", "exploratory", "claim", "retrieval", "unknown"}
)
ALLOWED_SOURCE_TYPES = frozenset(
    {
        "human_authored",
        "synthetic_generated",
        "synthetic_translated",
        "real_anonymized",
    }
)
ALLOWED_REVIEW_STATUSES = frozenset(
    {"unreviewed", "policy_reviewed", "native_reviewed"}
)
ALLOWED_SPLITS = frozenset({"train", "validation", "calibration", "test"})

REQUIRED_FIELDS = (
    "id",
    "language",
    "text",
    "intent",
    "template_id",
    "source_type",
    "review_status",
    "split",
)

_ELIGIBLE_INTENTS = frozenset({"question", "exploratory", "claim"})


class DatasetValidationError(ValueError):
    """Raised when a dataset file violates the format or leakage rules."""


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    id: str
    language: str
    text: str
    intent: str
    template_id: str
    source_type: str
    review_status: str
    split: str

    @property
    def eligible(self) -> bool:
        return self.intent in _ELIGIBLE_INTENTS

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> DatasetRecord:
        for field in REQUIRED_FIELDS:
            if field not in row:
                raise DatasetValidationError(f"missing field '{field}'")

        record_id = row["id"]
        if not isinstance(record_id, str) or not record_id.strip():
            raise DatasetValidationError("'id' must be a non-empty string")

        language = row["language"]
        if language not in SUPPORTED_LANGUAGE_CODES:
            raise DatasetValidationError(f"unsupported language '{language}'")

        text = row["text"]
        if not isinstance(text, str) or not text.strip():
            raise DatasetValidationError("'text' must be a non-empty string")

        intent = row["intent"]
        if intent not in ALLOWED_INTENTS:
            raise DatasetValidationError(f"unsupported intent '{intent}'")

        template_id = row["template_id"]
        if not isinstance(template_id, str) or not template_id.strip():
            raise DatasetValidationError("'template_id' must be a non-empty string")

        source_type = row["source_type"]
        if source_type not in ALLOWED_SOURCE_TYPES:
            raise DatasetValidationError(f"unsupported source_type '{source_type}'")

        review_status = row["review_status"]
        if review_status not in ALLOWED_REVIEW_STATUSES:
            raise DatasetValidationError(f"unsupported review_status '{review_status}'")

        split = row["split"]
        if split not in ALLOWED_SPLITS:
            raise DatasetValidationError(f"unsupported split '{split}'")

        return cls(
            id=record_id,
            language=language,
            text=text,
            intent=intent,
            template_id=template_id,
            source_type=source_type,
            review_status=review_status,
            split=split,
        )


def load_dataset(
    path: Path | str,
    *,
    require_acceptance_ready: bool = False,
) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    seen_ids: set[str] = set()
    template_splits: dict[str, set[str]] = {}

    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(
                f"line {line_number}: invalid JSON ({exc.msg})"
            ) from exc

        if not isinstance(row, dict):
            raise DatasetValidationError(
                f"line {line_number}: record must be a JSON object"
            )

        record = DatasetRecord.from_mapping(row)

        if record.id in seen_ids:
            raise DatasetValidationError(f"duplicate id '{record.id}'")
        seen_ids.add(record.id)

        template_splits.setdefault(record.template_id, set()).add(record.split)
        records.append(record)

    for template_id, splits in template_splits.items():
        if len(splits) > 1:
            raise DatasetValidationError(
                f"template '{template_id}' appears in multiple splits "
                f"({', '.join(sorted(splits))})"
            )

    if require_acceptance_ready:
        for record in records:
            if record.review_status != "native_reviewed":
                raise DatasetValidationError(
                    f"record '{record.id}' must be native_reviewed for an "
                    "acceptance dataset"
                )

    return records


def summarize(records: list[DatasetRecord]) -> dict[str, Any]:
    languages: dict[str, int] = {}
    splits: dict[str, int] = {}
    for record in records:
        languages[record.language] = languages.get(record.language, 0) + 1
        splits[record.split] = splits.get(record.split, 0) + 1
    return {
        "languages": dict(sorted(languages.items())),
        "records": len(records),
        "splits": dict(sorted(splits.items())),
    }
