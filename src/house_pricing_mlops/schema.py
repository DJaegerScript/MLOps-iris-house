"""Input and raw-dataset validation for the House Pricing model contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import pandas as pd

MODEL_SCHEMA_VERSION = "house-price-v1"
TARGET_NAME = "SalePrice"
ID_NAME = "Id"
NUMERIC_FEATURES = (
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
CATEGORICAL_FEATURES = ("Neighborhood", "KitchenQual", "CentralAir")
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RAW_REQUIRED_COLUMNS = (ID_NAME, *MODEL_FEATURES, TARGET_NAME)
OPTIONAL_PREDICTION_COLUMNS = (ID_NAME, TARGET_NAME)


class SchemaValidationError(ValueError):
    """A user-visible or dataset-level schema validation failure."""

    def __init__(self, *issues: str) -> None:
        self.issues = tuple(issue for issue in issues if issue)
        message = "; ".join(self.issues) or "invalid House Pricing schema"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ValidatedDataset:
    """Validated model features and target selected from raw training data."""

    features: pd.DataFrame
    target: pd.Series
    row_count: int
    raw_column_count: int


@dataclass(frozen=True, slots=True)
class BatchRowError:
    """Safe validation summary for one rejected batch row."""

    row_index: int
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BatchValidationResult:
    """Valid rows and safe diagnostics for a batch upload."""

    valid_frame: pd.DataFrame
    invalid_rows: tuple[BatchRowError, ...]

    @property
    def invalid_row_count(self) -> int:
        return len(self.invalid_rows)


def validate_raw_frame(frame: object) -> ValidatedDataset:
    """Validate a Kaggle-shaped training frame and select model columns.

    Extra Kaggle columns are intentionally allowed at intake time. They are
    excluded from the model feature frame after the required contract is
    checked.
    """

    if not isinstance(frame, pd.DataFrame):
        raise SchemaValidationError("training data must be a pandas DataFrame")

    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise SchemaValidationError(
            f"missing required columns: {', '.join(missing)}"
        )

    target = pd.to_numeric(frame[TARGET_NAME], errors="coerce")
    issues: list[str] = []
    if target.isna().any():
        issues.append("SalePrice must be numeric and finite")
    if not target.dropna().map(math.isfinite).all():
        issues.append("SalePrice must be numeric and finite")
    if (target.dropna() <= 0).any():
        issues.append("SalePrice must be positive")
    if issues:
        raise SchemaValidationError(*dict.fromkeys(issues))

    return ValidatedDataset(
        features=frame.loc[:, MODEL_FEATURES].copy(),
        target=target.rename(TARGET_NAME),
        row_count=len(frame),
        raw_column_count=len(frame.columns),
    )


def validate_single_features(features: object) -> dict[str, Any]:
    """Validate one exact model-feature mapping without filling missing input."""

    if not isinstance(features, Mapping):
        raise SchemaValidationError("features must be a mapping")

    expected = set(MODEL_FEATURES)
    actual = set(features)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    issues: list[str] = []
    if missing:
        issues.append(f"missing required features: {', '.join(missing)}")
    if unexpected:
        issues.append(f"unexpected features: {', '.join(unexpected)}")
    if issues:
        raise SchemaValidationError(*issues)

    validated: dict[str, Any] = {}
    for feature in MODEL_FEATURES:
        value = features[feature]
        if feature in NUMERIC_FEATURES:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise SchemaValidationError(f"{feature} must be numeric")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise SchemaValidationError(f"{feature} must be finite")
            validated[feature] = numeric_value
        else:
            if not isinstance(value, str) or not value.strip():
                raise SchemaValidationError(
                    f"{feature} must be a non-empty categorical value"
                )
            validated[feature] = value.strip()
    return validated


def validate_prediction_frame(
    frame: object, *, allow_labels: bool
) -> BatchValidationResult:
    """Validate exact batch columns and return valid rows plus safe row errors."""

    if not isinstance(frame, pd.DataFrame):
        raise SchemaValidationError("batch data must be a pandas DataFrame")

    allowed = set(MODEL_FEATURES) | {ID_NAME}
    if allow_labels:
        allowed.add(TARGET_NAME)
    unexpected = sorted(set(frame.columns) - allowed)
    missing = sorted(set(MODEL_FEATURES) - set(frame.columns))
    if unexpected:
        raise SchemaValidationError(
            f"unexpected columns: {', '.join(unexpected)}"
        )
    if missing:
        raise SchemaValidationError(
            f"missing required columns: {', '.join(missing)}"
        )

    valid_indices: list[Any] = []
    invalid_rows: list[BatchRowError] = []
    for row_index, row in frame.iterrows():
        row_values = {feature: row[feature] for feature in MODEL_FEATURES}
        try:
            validate_single_features(row_values)
            if allow_labels and TARGET_NAME in frame:
                _validate_label_value(row[TARGET_NAME])
        except SchemaValidationError as error:
            invalid_rows.append(BatchRowError(int(row_index), error.issues))
        else:
            valid_indices.append(row_index)

    valid_frame = frame.loc[valid_indices].copy()
    return BatchValidationResult(valid_frame, tuple(invalid_rows))


def _validate_label_value(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SchemaValidationError("SalePrice must be numeric and positive")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value <= 0:
        raise SchemaValidationError("SalePrice must be numeric and positive")


def feature_schema_dict() -> dict[str, object]:
    """Return a JSON-safe description of the declared model schema."""

    return {
        "version": MODEL_SCHEMA_VERSION,
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target": TARGET_NAME,
        "id_column": ID_NAME,
    }


__all__ = [
    "CATEGORICAL_FEATURES",
    "ID_NAME",
    "MODEL_FEATURES",
    "MODEL_SCHEMA_VERSION",
    "NUMERIC_FEATURES",
    "OPTIONAL_PREDICTION_COLUMNS",
    "RAW_REQUIRED_COLUMNS",
    "TARGET_NAME",
    "BatchRowError",
    "BatchValidationResult",
    "SchemaValidationError",
    "ValidatedDataset",
    "feature_schema_dict",
    "validate_prediction_frame",
    "validate_raw_frame",
    "validate_single_features",
]
