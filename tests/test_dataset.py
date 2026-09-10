import json
from pathlib import Path

import pytest

from eea_query_intent.dataset import DatasetValidationError, load_dataset


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def valid_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "en-0001",
        "language": "en",
        "text": "What causes air pollution?",
        "intent": "question",
        "template_id": "template-0001",
        "source_type": "human_authored",
        "review_status": "native_reviewed",
        "split": "test",
    }
    row.update(overrides)
    return row


def test_loads_a_valid_unicode_dataset(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    write_jsonl(
        path,
        [valid_row(language="ga", text="Cad is cúis le truailliú aeir?")],
    )

    records = load_dataset(path)

    assert records[0].language == "ga"
    assert records[0].eligible is True


def test_rejects_unsupported_languages(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    write_jsonl(path, [valid_row(language="xx")])

    with pytest.raises(DatasetValidationError, match="unsupported language 'xx'"):
        load_dataset(path)


def test_accepts_both_norwegian_written_standards(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    write_jsonl(
        path,
        [
            valid_row(id="nb-0001", language="nb", template_id="nb-template"),
            valid_row(id="nn-0001", language="nn", template_id="nn-template"),
        ],
    )

    assert [record.language for record in load_dataset(path)] == ["nb", "nn"]


def test_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    write_jsonl(path, [valid_row(), valid_row()])

    with pytest.raises(DatasetValidationError, match="duplicate id 'en-0001'"):
        load_dataset(path)


def test_rejects_translation_template_leakage_across_splits(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    write_jsonl(
        path,
        [
            valid_row(split="train"),
            valid_row(
                id="fr-0001",
                language="fr",
                text="Quelles sont les causes de la pollution de l’air ?",
                split="test",
            ),
        ],
    )

    with pytest.raises(
        DatasetValidationError,
        match="template 'template-0001' appears in multiple splits",
    ):
        load_dataset(path)


def test_acceptance_data_must_be_native_reviewed(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    write_jsonl(path, [valid_row(review_status="policy_reviewed")])

    with pytest.raises(DatasetValidationError, match="must be native_reviewed"):
        load_dataset(path, require_acceptance_ready=True)
