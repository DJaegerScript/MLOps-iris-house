"""Educational batch input drift profiling and PSI-style scoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from house_pricing_mlops.schema import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
)

DRIFT_WARNING_THRESHOLD = 0.20
MIN_RELIABLE_BATCH_SIZE = 30
_MISSING_CATEGORY = "__MISSING__"
_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Safe drift diagnostics for one batch."""

    per_feature_scores: Mapping[str, float]
    drifted_features: tuple[str, ...]
    unknown_category_count: int
    invalid_row_count: int
    sample_size_warning: bool
    threshold: float = DRIFT_WARNING_THRESHOLD

    @property
    def number_drifted_features(self) -> int:
        return len(self.drifted_features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_feature_scores": dict(self.per_feature_scores),
            "drifted_features": list(self.drifted_features),
            "number_drifted_features": self.number_drifted_features,
            "unknown_category_count": self.unknown_category_count,
            "invalid_row_count": self.invalid_row_count,
            "sample_size_warning": self.sample_size_warning,
            "threshold": self.threshold,
        }


def build_reference_profile(features: object) -> dict[str, Any]:
    """Build JSON-safe statistics from training features."""

    frame = _require_exact_features(features)
    profile: dict[str, Any] = {
        "schema_version": "house-price-v1",
        "sample_count": len(frame),
        "numeric": {},
        "categorical": {},
        "drift_warning_threshold": DRIFT_WARNING_THRESHOLD,
    }
    for feature in NUMERIC_FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce").dropna().to_numpy()
        if len(values) == 0:
            raise ValueError(f"numeric feature has no usable values: {feature}")
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if minimum == maximum:
            edges = np.array([minimum - 0.5, maximum + 0.5])
        else:
            edges = np.linspace(minimum, maximum, num=11)
            edges[-1] = np.nextafter(edges[-1], np.inf)
        counts, _ = np.histogram(np.clip(values, edges[0], edges[-1]), bins=edges)
        distribution = _normalize(counts.astype(float))
        profile["numeric"][feature] = {
            "minimum": minimum,
            "maximum": maximum,
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
            "quantiles": {
                str(q): float(np.quantile(values, q))
                for q in (0.0, 0.25, 0.5, 0.75, 1.0)
            },
            "histogram_bins": [float(edge) for edge in edges],
            "histogram_distribution": distribution,
        }
    for feature in CATEGORICAL_FEATURES:
        values = frame[feature].fillna(_MISSING_CATEGORY).astype(str)
        counts = values.value_counts(normalize=True, sort=False)
        distribution = {
            str(category): float(counts[category])
            for category in sorted(counts.index.astype(str))
        }
        profile["categorical"][feature] = {
            "distribution": distribution,
            "categories": sorted(distribution),
            "unknown_category_rate": 0.0,
        }
    return profile


def score_batch(
    reference_profile: Mapping[str, Any],
    features: object,
    *,
    invalid_row_count: int = 0,
    threshold: float = DRIFT_WARNING_THRESHOLD,
) -> DriftReport:
    """Score a batch against a reference profile using an educational PSI rule."""

    frame = _require_exact_features(features)
    if invalid_row_count < 0:
        raise ValueError("invalid_row_count must be non-negative")
    scores: dict[str, float] = {}
    unknown_category_count = 0
    for feature in NUMERIC_FEATURES:
        config = reference_profile["numeric"][feature]
        values = pd.to_numeric(frame[feature], errors="coerce").dropna().to_numpy()
        edges = np.asarray(config["histogram_bins"], dtype=float)
        clipped = np.clip(values, edges[0], edges[-1])
        counts, _ = np.histogram(clipped, bins=edges)
        actual = _normalize(counts.astype(float))
        expected = [float(value) for value in config["histogram_distribution"]]
        scores[feature] = _psi(expected, actual)
    for feature in CATEGORICAL_FEATURES:
        config = reference_profile["categorical"][feature]
        expected_distribution = {
            str(key): float(value)
            for key, value in config["distribution"].items()
        }
        categories = list(expected_distribution)
        values = frame[feature].fillna(_MISSING_CATEGORY).astype(str)
        unknown = ~values.isin(categories)
        unknown_category_count += int(unknown.sum())
        actual_distribution = [
            float((values == category).sum()) for category in categories
        ]
        if unknown.any():
            categories.append("__UNKNOWN__")
            actual_distribution.append(float(unknown.sum()))
            expected_distribution["__UNKNOWN__"] = 0.0
        scores[feature] = _psi(
            [expected_distribution[category] for category in categories],
            _normalize(np.asarray(actual_distribution)),
        )
    drifted = tuple(
        feature for feature in MODEL_FEATURES if scores[feature] >= threshold
    )
    return DriftReport(
        per_feature_scores=scores,
        drifted_features=drifted,
        unknown_category_count=unknown_category_count,
        invalid_row_count=invalid_row_count,
        sample_size_warning=len(frame) < MIN_RELIABLE_BATCH_SIZE,
        threshold=threshold,
    )


def _require_exact_features(features: object) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame):
        raise ValueError("drift features must be a pandas DataFrame")
    missing = sorted(set(MODEL_FEATURES) - set(features.columns))
    unexpected = sorted(set(features.columns) - set(MODEL_FEATURES))
    if missing:
        raise ValueError(f"missing required features: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"unexpected features: {', '.join(unexpected)}")
    return features.loc[:, MODEL_FEATURES].copy()


def _normalize(values: np.ndarray) -> list[float]:
    total = float(values.sum())
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [float(value / total) for value in values]


def _psi(expected: list[float], actual: list[float]) -> float:
    expected_values = np.asarray(expected, dtype=float)
    actual_values = np.asarray(actual, dtype=float)
    expected_values = (expected_values + _EPSILON) / (
        expected_values + _EPSILON
    ).sum()
    actual_values = (actual_values + _EPSILON) / (actual_values + _EPSILON).sum()
    return float(
        np.sum(
            (actual_values - expected_values)
            * np.log(actual_values / expected_values)
        )
    )


__all__ = [
    "DRIFT_WARNING_THRESHOLD",
    "MIN_RELIABLE_BATCH_SIZE",
    "DriftReport",
    "build_reference_profile",
    "score_batch",
]
