"""Command line interface for dataset validation and benchmark evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eea_query_intent.dataset import (
    DatasetValidationError,
    load_dataset,
    summarize,
)
from eea_query_intent.metrics import (
    AcceptanceThresholds,
    PredictionRecord,
    PredictionValidationError,
    evaluate_predictions,
)


def _emit(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _fail(message: str) -> int:
    print(json.dumps({"error": message}, ensure_ascii=False), file=sys.stderr)
    return 2


def _load_predictions(path: Path) -> list[PredictionRecord]:
    predictions: list[PredictionRecord] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PredictionValidationError(
                f"line {line_number}: invalid JSON ({exc.msg})"
            ) from exc
        if not isinstance(row, dict):
            raise PredictionValidationError(
                f"line {line_number}: prediction must be a JSON object"
            )
        predictions.append(PredictionRecord.from_mapping(row))
    return predictions


def _command_data_validate(args: argparse.Namespace) -> int:
    try:
        records = load_dataset(
            args.path,
            require_acceptance_ready=args.require_acceptance_ready,
        )
    except (DatasetValidationError, OSError) as exc:
        return _fail(str(exc))
    _emit(summarize(records))
    return 0


def _command_evaluate(args: argparse.Namespace) -> int:
    try:
        gold = load_dataset(args.gold)
        predictions = _load_predictions(Path(args.predictions))
    except (DatasetValidationError, PredictionValidationError, OSError) as exc:
        return _fail(str(exc))

    thresholds = AcceptanceThresholds(
        minimum_eligible_count=args.minimum_eligible_count,
        minimum_no_ai_count=args.minimum_no_ai_count,
        maximum_no_ai_false_positive_rate=args.max_no_ai_false_positive_rate,
        minimum_eligible_precision=args.min_eligible_precision,
        minimum_macro_f1=args.min_macro_f1,
    )

    try:
        report = evaluate_predictions(gold, predictions, thresholds)
    except ValueError as exc:
        return _fail(str(exc))

    _emit(report)
    return 0 if report["passes"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eea-query-intent",
        description="Benchmark tooling for EEA search-query intent routing.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    data = subcommands.add_parser("data", help="Dataset utilities")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    validate = data_sub.add_parser(
        "validate", help="Validate a dataset and print its summary"
    )
    validate.add_argument("path", type=Path)
    validate.add_argument(
        "--require-acceptance-ready",
        action="store_true",
        help="Require every record to be native_reviewed",
    )
    validate.set_defaults(func=_command_data_validate)

    evaluate = subcommands.add_parser(
        "evaluate", help="Score predictions against a gold dataset"
    )
    evaluate.add_argument("--gold", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--minimum-eligible-count", type=int, default=300)
    evaluate.add_argument("--minimum-no-ai-count", type=int, default=300)
    evaluate.add_argument("--max-no-ai-false-positive-rate", type=float, default=0.01)
    evaluate.add_argument("--min-eligible-precision", type=float, default=0.98)
    evaluate.add_argument("--min-macro-f1", type=float, default=0.95)
    evaluate.set_defaults(func=_command_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
