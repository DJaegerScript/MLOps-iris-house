"""Reproducible House Pricing training entry point."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from house_pricing_mlops.provenance import build_dataset_provenance
from house_pricing_mlops.schema import SchemaValidationError
from house_pricing_mlops.tracking import MLflowTracker
from house_pricing_mlops.training import TrainingConfig, train_house_price_model


class TrainingInputError(ValueError):
    """A safe, actionable input error for the training command."""


def run_training(
    input_path: str | Path,
    *,
    dataset_version: str,
    output_dir: str | Path,
    tracking_uri: str | None = None,
    git_commit_sha: str | None = None,
    random_seed: int = 42,
) -> dict[str, object]:
    """Train the fixture or real dataset and write a safe JSON summary."""

    dataset_path = Path(input_path)
    if not dataset_path.is_file():
        raise TrainingInputError("dataset file does not exist")
    if dataset_path.name != "train.csv":
        raise TrainingInputError("training input must be named train.csv")

    try:
        frame = pd.read_csv(dataset_path)
        config = TrainingConfig(random_seed=random_seed)
        provenance = build_dataset_provenance(
            dataset_path,
            dataset_version=dataset_version,
            git_commit_sha=git_commit_sha,
            training_configuration=config.to_dict(),
        )
    except (OSError, pd.errors.ParserError, SchemaValidationError, ValueError) as error:
        raise TrainingInputError(f"invalid training dataset: {error}") from error

    started = time.perf_counter()
    result = train_house_price_model(frame, config=config)
    duration = time.perf_counter() - started
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tracker_uri = tracking_uri or (output_path.parent / "mlruns").resolve().as_uri()
    tracked = MLflowTracker(tracker_uri).log(
        result,
        provenance,
        training_duration_seconds=duration,
    )
    summary: dict[str, object] = {
        "dataset_version": provenance.dataset_version,
        "dataset_sha256": provenance.dataset_sha256,
        "git_commit_sha": provenance.git_commit_sha,
        "selected_model": result.selected_model_name,
        "metrics": result.test_metrics,
        "candidate_validation_metrics": {
            candidate.name: candidate.validation_metrics
            for candidate in result.candidates
        },
        "training_duration_seconds": duration,
        "mlflow_run_id": tracked.run_id,
        "mlflow_experiment_id": tracked.experiment_id,
        "tracking_uri": tracked.tracking_uri,
        "random_seed": config.random_seed,
    }
    (output_path / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--git-commit-sha")
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_training(
            args.input,
            dataset_version=args.dataset_version,
            output_dir=args.output_dir,
            tracking_uri=args.tracking_uri,
            git_commit_sha=args.git_commit_sha,
            random_seed=args.random_seed,
        )
    except TrainingInputError as error:
        print(f"training failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
