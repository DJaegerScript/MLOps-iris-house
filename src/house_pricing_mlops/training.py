"""Deterministic, CPU-friendly training for the House Pricing model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from house_pricing_mlops.schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    validate_raw_frame,
)

CANDIDATE_MODEL_NAMES = ("dummy", "random_forest", "gradient_boosting")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Reproducible workshop-sized training configuration."""

    random_seed: int = 42
    validation_size: float = 0.15
    test_size: float = 0.15
    random_forest_estimators: int = 40
    gradient_boosting_estimators: int = 80

    def __post_init__(self) -> None:
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed must be a non-negative integer")
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if self.validation_size + self.test_size >= 1:
            raise ValueError("validation_size plus test_size must be below 1")
        if self.random_forest_estimators < 1:
            raise ValueError("random_forest_estimators must be positive")
        if self.gradient_boosting_estimators < 1:
            raise ValueError("gradient_boosting_estimators must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "random_seed": self.random_seed,
            "validation_size": self.validation_size,
            "test_size": self.test_size,
            "random_forest_estimators": self.random_forest_estimators,
            "gradient_boosting_estimators": self.gradient_boosting_estimators,
            "target_transform": "log1p_expm1",
        }


@dataclass(frozen=True, slots=True)
class SplitIndices:
    """Original input indices for the three deterministic partitions."""

    train_indices: tuple[Any, ...]
    validation_indices: tuple[Any, ...]
    test_indices: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class CandidateTrainingResult:
    """One fitted candidate and its validation/test evidence."""

    name: str
    pipeline: TransformedTargetRegressor
    parameters: dict[str, object]
    validation_metrics: dict[str, float | None]
    test_metrics: dict[str, float | None]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """All artifacts needed for selection, evaluation, and bundle creation."""

    candidates: tuple[CandidateTrainingResult, ...]
    selected_model_name: str
    split: SplitIndices
    selected_features: pd.DataFrame
    training_features: pd.DataFrame
    training_target: pd.Series
    test_metrics: dict[str, float | None]
    config: TrainingConfig

    @property
    def selected_candidate(self) -> CandidateTrainingResult:
        return next(
            candidate
            for candidate in self.candidates
            if candidate.name == self.selected_model_name
        )

    @property
    def selected_pipeline(self) -> TransformedTargetRegressor:
        return self.selected_candidate.pipeline


def calculate_metrics(
    actual: object, predicted: object
) -> dict[str, float | None]:
    """Calculate metrics in original price units."""

    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    rmse = math.sqrt(mean_squared_error(actual_values, predicted_values))
    rmsle: float | None = None
    if np.all(actual_values >= 0) and np.all(predicted_values >= 0):
        rmsle = math.sqrt(mean_squared_log_error(actual_values, predicted_values))
    return {
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(rmse),
        "r2": float(r2_score(actual_values, predicted_values)),
        "rmsle": rmsle,
    }


def train_house_price_model(
    frame: pd.DataFrame, *, config: TrainingConfig | None = None
) -> TrainingResult:
    """Fit candidates without using the test partition for selection or fitting."""

    training_config = config or TrainingConfig()
    validated = validate_raw_frame(frame)
    features = validated.features
    target = validated.target

    train_indices, remainder_indices = train_test_split(
        features.index.to_numpy(),
        test_size=training_config.validation_size + training_config.test_size,
        random_state=training_config.random_seed,
    )
    validation_fraction = training_config.test_size / (
        training_config.validation_size + training_config.test_size
    )
    validation_indices, test_indices = train_test_split(
        remainder_indices,
        test_size=validation_fraction,
        random_state=training_config.random_seed,
    )
    split = SplitIndices(
        tuple(train_indices.tolist()),
        tuple(validation_indices.tolist()),
        tuple(test_indices.tolist()),
    )

    train_features = features.loc[list(split.train_indices)]
    validation_features = features.loc[list(split.validation_indices)]
    test_features = features.loc[list(split.test_indices)]
    train_target = target.loc[list(split.train_indices)]
    validation_target = target.loc[list(split.validation_indices)]
    test_target = target.loc[list(split.test_indices)]

    candidates: list[CandidateTrainingResult] = []
    for name, estimator, parameters in _candidate_estimators(training_config):
        pipeline = _build_pipeline(estimator)
        pipeline.fit(train_features, train_target)
        validation_predictions = pipeline.predict(validation_features)
        validation_metrics = calculate_metrics(
            validation_target, validation_predictions
        )
        candidates.append(
            CandidateTrainingResult(
                name=name,
                pipeline=pipeline,
                parameters=parameters,
                validation_metrics=validation_metrics,
                test_metrics={},
            )
        )

    selected = min(
        candidates,
        key=lambda candidate: (
            candidate.validation_metrics["rmse"]
            if candidate.validation_metrics["rmse"] is not None
            else float("inf"),
            CANDIDATE_MODEL_NAMES.index(candidate.name),
        ),
    )
    test_predictions = selected.pipeline.predict(test_features)
    test_metrics = calculate_metrics(test_target, test_predictions)
    candidates = [
        CandidateTrainingResult(
            name=candidate.name,
            pipeline=candidate.pipeline,
            parameters=candidate.parameters,
            validation_metrics=candidate.validation_metrics,
            test_metrics=test_metrics if candidate.name == selected.name else {},
        )
        for candidate in candidates
    ]

    return TrainingResult(
        candidates=tuple(candidates),
        selected_model_name=selected.name,
        split=split,
        selected_features=features.copy(),
        training_features=train_features.copy(),
        training_target=train_target.copy(),
        test_metrics=test_metrics,
        config=training_config,
    )


def _build_pipeline(estimator: Any) -> TransformedTargetRegressor:
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__MISSING__"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )
    regressor = Pipeline(
        [("preprocessor", preprocessor), ("model", estimator)]
    )
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def _candidate_estimators(
    config: TrainingConfig,
) -> list[tuple[str, Any, dict[str, object]]]:
    return [
        (
            "dummy",
            DummyRegressor(strategy="mean"),
            {"strategy": "mean"},
        ),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=config.random_forest_estimators,
                random_state=config.random_seed,
                n_jobs=1,
            ),
            {
                "n_estimators": config.random_forest_estimators,
                "random_state": config.random_seed,
                "n_jobs": 1,
            },
        ),
        (
            "gradient_boosting",
            GradientBoostingRegressor(
                n_estimators=config.gradient_boosting_estimators,
                random_state=config.random_seed,
            ),
            {
                "n_estimators": config.gradient_boosting_estimators,
                "random_state": config.random_seed,
            },
        ),
    ]


__all__ = [
    "CANDIDATE_MODEL_NAMES",
    "CandidateTrainingResult",
    "SplitIndices",
    "TrainingConfig",
    "TrainingResult",
    "calculate_metrics",
    "train_house_price_model",
]
