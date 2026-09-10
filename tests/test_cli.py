import json
from pathlib import Path

from eea_query_intent.cli import main


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_validate_data_command_reports_dataset_summary(tmp_path: Path, capsys) -> None:
    dataset = tmp_path / "gold.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "id": "en-1",
                "language": "en",
                "text": "What causes air pollution?",
                "intent": "question",
                "template_id": "template-1",
                "source_type": "human_authored",
                "review_status": "native_reviewed",
                "split": "test",
            }
        ],
    )

    exit_code = main(["data", "validate", str(dataset)])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "languages": {"en": 1},
        "records": 1,
        "splits": {"test": 1},
    }


def test_evaluate_command_returns_nonzero_when_a_language_fails(
    tmp_path: Path, capsys
) -> None:
    gold = tmp_path / "gold.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        gold,
        [
            {
                "id": "mt-1",
                "language": "mt",
                "text": "kwalità tal-arja",
                "intent": "retrieval",
                "template_id": "template-1",
                "source_type": "human_authored",
                "review_status": "native_reviewed",
                "split": "test",
            }
        ],
    )
    write_jsonl(
        predictions,
        [
            {
                "id": "mt-1",
                "intent": "question",
                "eligible": True,
                "eligible_probability": 0.99,
                "confidence": 0.99,
                "abstained": False,
                "model_version": "candidate",
                "latency_ms": 12.5,
            }
        ],
    )

    exit_code = main(
        [
            "evaluate",
            "--gold",
            str(gold),
            "--predictions",
            str(predictions),
            "--minimum-eligible-count",
            "0",
            "--minimum-no-ai-count",
            "1",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["passes"] is False
    assert report["failing_languages"] == ["mt"]
