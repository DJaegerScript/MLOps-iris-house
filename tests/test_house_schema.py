"""Tests for the House Pricing raw and prediction schemas."""

from pathlib import Path

import pandas as pd
import pytest

from house_pricing_mlops.schema import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_NAME,
    SchemaValidationError,
    validate_prediction_frame,
    validate_raw_frame,
    validate_single_features,
)

FIXTURE_PATH = Path("tests/fixtures/house_prices/train.csv")


def _fixture_frame() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_PATH)


def test_declared_feature_partition_and_target_are_exact() -> None:
    assert NUMERIC_FEATURES == (
        "OverallQual",
        "GrLivArea",
        "GarageCars",
        "TotalBsmtSF",
        "1stFlrSF",
        "YearBuilt",
        "FullBath",
        "TotRmsAbvGrd",
        "GarageArea",
    )
    assert CATEGORICAL_FEATURES == ("Neighborhood", "KitchenQual", "CentralAir")
    assert MODEL_FEATURES == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert TARGET_NAME == "SalePrice"


def test_raw_validation_removes_id_and_selects_declared_features() -> None:
    frame = _fixture_frame().assign(UnusedKaggleColumn="ignored")

    validated = validate_raw_frame(frame)

    assert tuple(validated.features.columns) == MODEL_FEATURES
    assert validated.target.name == TARGET_NAME
    assert validated.target.tolist()[0] == 208500
    assert validated.row_count == len(frame)


def test_raw_validation_requires_target() -> None:
    frame = _fixture_frame().drop(columns=[TARGET_NAME])

    with pytest.raises(SchemaValidationError, match="SalePrice"):
        validate_raw_frame(frame)


@pytest.mark.parametrize("value", [0, -1, float("nan"), "not-a-price"])
def test_raw_validation_rejects_invalid_target_values(value: object) -> None:
    frame = _fixture_frame()
    frame[TARGET_NAME] = frame[TARGET_NAME].astype(object)
    frame.loc[0, TARGET_NAME] = value

    with pytest.raises(SchemaValidationError, match="SalePrice"):
        validate_raw_frame(frame)


def test_raw_validation_requires_all_model_columns() -> None:
    frame = _fixture_frame().drop(columns=["GarageArea"])

    with pytest.raises(SchemaValidationError, match="GarageArea"):
        validate_raw_frame(frame)


def test_prediction_validation_accepts_model_features_and_optional_columns() -> None:
    frame = _fixture_frame().loc[:1, list(MODEL_FEATURES) + ["Id", TARGET_NAME]]

    result = validate_prediction_frame(frame, allow_labels=True)

    assert result.valid_frame.equals(frame)
    assert result.invalid_row_count == 0
    assert result.invalid_rows == ()


def test_prediction_validation_rejects_unexpected_columns() -> None:
    frame = _fixture_frame().loc[:0, list(MODEL_FEATURES)].assign(Address="private")

    with pytest.raises(SchemaValidationError, match="unexpected"):
        validate_prediction_frame(frame, allow_labels=False)


def test_single_validation_rejects_missing_and_non_finite_values() -> None:
    values = {feature: 1.0 for feature in NUMERIC_FEATURES}
    values.update({feature: "known" for feature in CATEGORICAL_FEATURES})
    values.pop("GarageArea")

    with pytest.raises(SchemaValidationError, match="GarageArea"):
        validate_single_features(values)

    values["GarageArea"] = float("inf")
    with pytest.raises(SchemaValidationError, match="GarageArea"):
        validate_single_features(values)


def test_single_validation_returns_declared_order_without_extra_values() -> None:
    values = {feature: 1.0 for feature in NUMERIC_FEATURES}
    values.update({feature: "known" for feature in CATEGORICAL_FEATURES})

    validated = validate_single_features(values)

    assert tuple(validated) == MODEL_FEATURES
