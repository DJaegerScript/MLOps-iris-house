"""Tests for the offline House Prices dataset intake boundary."""

from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from scripts.intake_house_price_dataset import (
    FIXTURE_SOURCE_URL,
    IntakeError,
    intake_dataset,
)

FIXTURE_PATH = Path("tests/fixtures/house_prices/train.csv")


def test_intake_records_provenance_and_immutable_s3_key(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    source.write_bytes(FIXTURE_PATH.read_bytes())

    result = intake_dataset(
        source,
        dataset_version="v1",
        output_dir=tmp_path / "intake",
        download_timestamp="2026-09-21T00:00:00+00:00",
        git_commit_sha="abc123",
    )

    assert result["source_url"] == "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data"
    assert result["dataset_version"] == "v1"
    assert result["download_timestamp"] == "2026-09-21T00:00:00+00:00"
    assert result["row_count"] == 12
    assert result["column_count"] == 14
    assert result["target_column"] == "SalePrice"
    assert result["git_commit_sha"] == "abc123"
    assert result["s3_object"] == {
        "bucket": None,
        "key": "datasets/house-prices/v1/train.csv",
    }
    assert (tmp_path / "intake" / "provenance.json").is_file()


def test_fixture_intake_uses_explicit_non_production_source_marker(
    tmp_path: Path,
) -> None:
    result = intake_dataset(
        FIXTURE_PATH,
        dataset_version="fixture-v1",
        output_dir=tmp_path,
    )

    assert result["source_url"] == FIXTURE_SOURCE_URL


def test_intake_accepts_kaggle_zip_with_exact_train_filename(tmp_path: Path) -> None:
    archive = tmp_path / "house-prices.zip"
    with ZipFile(archive, "w") as zip_file:
        zip_file.write(FIXTURE_PATH, "download/train.csv")

    result = intake_dataset(
        archive,
        dataset_version="v1",
        output_dir=tmp_path / "intake",
        source_url="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data",
    )

    assert result["input_type"] == "kaggle_zip"
    assert result["row_count"] == 12


def test_intake_rejects_non_train_csv_filename(tmp_path: Path) -> None:
    source = tmp_path / "test.csv"
    source.write_bytes(FIXTURE_PATH.read_bytes())

    with pytest.raises(IntakeError, match="named train.csv"):
        intake_dataset(source, dataset_version="v1", output_dir=tmp_path / "out")


def test_intake_rejects_duplicate_target_column(tmp_path: Path) -> None:
    frame = pd.read_csv(FIXTURE_PATH)
    frame.insert(
        len(frame.columns),
        "SalePrice",
        frame["SalePrice"],
        allow_duplicates=True,
    )
    source = tmp_path / "train.csv"
    frame.to_csv(source, index=False)

    with pytest.raises(IntakeError, match="duplicate columns.*SalePrice"):
        intake_dataset(source, dataset_version="v1", output_dir=tmp_path / "out")


def test_intake_rejects_invalid_target_values(tmp_path: Path) -> None:
    frame = pd.read_csv(FIXTURE_PATH)
    frame.loc[0, "SalePrice"] = 0
    source = tmp_path / "train.csv"
    frame.to_csv(source, index=False)

    with pytest.raises(IntakeError, match="SalePrice"):
        intake_dataset(source, dataset_version="v1", output_dir=tmp_path / "out")


def test_intake_reports_missing_input_without_fabricating_results(
    tmp_path: Path,
) -> None:
    with pytest.raises(IntakeError, match="does not exist"):
        intake_dataset(
            tmp_path / "missing" / "train.csv",
            dataset_version="v1",
            output_dir=tmp_path / "out",
        )
