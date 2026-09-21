"""Tests for the reproducible House Pricing training entry point."""

from pathlib import Path

import pytest

from scripts.train_house_price_model import TrainingInputError, run_training

FIXTURE_PATH = Path("tests/fixtures/house_prices/train.csv")


def test_training_fails_clearly_when_dataset_is_missing(tmp_path: Path) -> None:
    with pytest.raises(TrainingInputError, match="dataset file does not exist"):
        run_training(
            tmp_path / "missing.csv",
            dataset_version="v1",
            output_dir=tmp_path / "output",
        )


def test_training_writes_safe_summary_with_dataset_and_run_provenance(
    tmp_path: Path,
) -> None:
    summary = run_training(
        FIXTURE_PATH,
        dataset_version="fixture-v1",
        output_dir=tmp_path / "output",
        tracking_uri=(tmp_path / "mlruns").as_uri(),
        git_commit_sha="abc123",
        random_seed=11,
    )

    assert summary["dataset_version"] == "fixture-v1"
    assert summary["git_commit_sha"] == "abc123"
    assert summary["selected_model"]
    assert summary["mlflow_run_id"]
    assert summary["metrics"]["rmse"] is not None
    assert summary["model_name"] == "house-price-model"
    assert summary["model_version"] == "v1"
    assert summary["bundle_sha256"]
    assert (tmp_path / "output" / "model.tar.gz").is_file()
    assert (tmp_path / "output" / "training_summary.json").is_file()


def test_training_rejects_non_csv_input(tmp_path: Path) -> None:
    input_path = tmp_path / "train.txt"
    input_path.write_text("not,csv\n", encoding="utf-8")

    with pytest.raises(TrainingInputError, match="train.csv"):
        run_training(
            input_path,
            dataset_version="v1",
            output_dir=tmp_path / "output",
        )
