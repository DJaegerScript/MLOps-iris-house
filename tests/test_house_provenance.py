"""Tests for dataset provenance and checksum capture."""

from pathlib import Path

import pandas as pd

from house_pricing_mlops.provenance import (
    DEFAULT_DATASET_SOURCE_URL,
    build_dataset_provenance,
    sha256_file,
)

FIXTURE_PATH = Path("tests/fixtures/house_prices/train.csv")


def test_sha256_file_matches_known_bytes(tmp_path: Path) -> None:
    path = tmp_path / "dataset.csv"
    path.write_bytes(b"house-prices")

    assert sha256_file(path) == (
        "ed0103f62dc1a6fd53de9e71b759c7749224a5513872dca4bf9ed0d53cc92834"
    )


def test_build_dataset_provenance_records_required_fields() -> None:
    provenance = build_dataset_provenance(
        FIXTURE_PATH,
        dataset_version="fixture-v1",
        download_timestamp="2026-09-21T00:00:00+00:00",
        git_commit_sha="abc123",
        training_configuration={"seed": 42},
    )

    assert provenance.source_url == DEFAULT_DATASET_SOURCE_URL
    assert provenance.dataset_version == "fixture-v1"
    assert provenance.download_timestamp == "2026-09-21T00:00:00+00:00"
    assert provenance.dataset_sha256 == sha256_file(FIXTURE_PATH)
    assert provenance.row_count == len(pd.read_csv(FIXTURE_PATH))
    assert provenance.column_count == len(pd.read_csv(FIXTURE_PATH).columns)
    assert provenance.target_column == "SalePrice"
    assert provenance.git_commit_sha == "abc123"
    assert provenance.training_configuration == {"seed": 42}
    assert provenance.feature_schema["version"] == "house-price-v1"


def test_provenance_serializes_to_json_safe_mapping() -> None:
    provenance = build_dataset_provenance(
        FIXTURE_PATH,
        dataset_version="fixture-v1",
        download_timestamp="2026-09-21T00:00:00+00:00",
        git_commit_sha="abc123",
        training_configuration={},
    )

    payload = provenance.to_dict()

    assert payload["dataset_sha256"] == provenance.dataset_sha256
    assert payload["feature_schema"]["numeric_features"]
    assert payload["feature_schema"]["categorical_features"]
