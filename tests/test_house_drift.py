"""Tests for House Pricing reference profiles and educational drift scores."""

from pathlib import Path

import pandas as pd
import pytest

from house_pricing_mlops.drift import build_reference_profile, score_batch
from house_pricing_mlops.schema import MODEL_FEATURES

FIXTURE_PATH = Path("tests/fixtures/house_prices/train.csv")


def _features() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_PATH).loc[:, MODEL_FEATURES]


def test_reference_profile_records_numeric_and_categorical_statistics() -> None:
    profile = build_reference_profile(_features())

    assert profile["sample_count"] == 12
    assert set(profile["numeric"]) == {
        "OverallQual",
        "GrLivArea",
        "GarageCars",
        "TotalBsmtSF",
        "1stFlrSF",
        "YearBuilt",
        "FullBath",
        "TotRmsAbvGrd",
        "GarageArea",
    }
    assert profile["numeric"]["GrLivArea"]["minimum"] == 1040.0
    assert profile["numeric"]["GrLivArea"]["maximum"] == 1960.0
    assert profile["numeric"]["GrLivArea"]["histogram_bins"]
    assert profile["categorical"]["Neighborhood"]["distribution"]
    assert profile["categorical"]["Neighborhood"]["unknown_category_rate"] == 0.0


def test_unchanged_batch_has_no_drifted_features() -> None:
    features = _features()
    report = score_batch(build_reference_profile(features), features)

    assert report.number_drifted_features == 0
    assert report.drifted_features == ()
    assert report.unknown_category_count == 0
    assert report.invalid_row_count == 0


def test_shifted_batch_reports_numeric_and_categorical_drift() -> None:
    reference = _features()
    shifted = reference.copy()
    shifted["GrLivArea"] = 5000
    shifted["Neighborhood"] = "UnknownNeighborhood"

    report = score_batch(build_reference_profile(reference), shifted)

    assert report.number_drifted_features >= 1
    assert report.per_feature_scores["GrLivArea"] > 0
    assert "Neighborhood" in report.drifted_features
    assert report.unknown_category_count == len(shifted)


def test_invalid_rows_and_small_batch_warning_are_reported() -> None:
    report = score_batch(
        build_reference_profile(_features()),
        _features().iloc[:1],
        invalid_row_count=2,
    )

    assert report.invalid_row_count == 2
    assert report.sample_size_warning is True


def test_profile_rejects_unexpected_columns() -> None:
    with pytest.raises(ValueError, match="unexpected"):
        build_reference_profile(_features().assign(Address="private"))
