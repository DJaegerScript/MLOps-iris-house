"""Validated prediction service for the pre-trained Iris model."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

from iris_mlops.manifest import ModelManifest
from iris_mlops.model import IrisPrediction, LoadedIrisModel


class PredictionInputError(ValueError):
    """Stable public error raised when input does not match the model schema."""

    def __init__(self, reason: str = "invalid feature schema") -> None:
        self.reason = reason
        super().__init__("Invalid prediction input")


class PredictionLogger(Protocol):
    """Logging boundary used by the prediction service."""

    def invalid_input(self) -> None:
        """Log rejected input."""

    def prediction_success(
        self,
        *,
        predicted_class: str,
        confidence: float,
        latency_ms: float,
        valid_input: bool,
    ) -> None:
        """Log a successful prediction."""

    def prediction_error(self, *, valid_input: bool, error: str) -> None:
        """Log a prediction failure."""

    def metric(self, name: str, value: float) -> None:
        """Log a CloudWatch metric."""


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Public result returned by the prediction service."""

    predicted_class: str
    confidence: float
    probabilities: Mapping[str, float]
    latency_ms: float
    valid_input: bool


def validate_features(
    features: object, manifest: ModelManifest
) -> tuple[float, ...]:
    """Validate exactly the manifest's ordered four-feature input schema."""

    expected_features = tuple(manifest.features)
    if len(expected_features) != 4 or not isinstance(features, Mapping):
        raise PredictionInputError("features must be a mapping of four values")
    if set(features) != set(expected_features):
        raise PredictionInputError("feature names do not match the manifest")

    validated: list[float] = []
    for feature_name in expected_features:
        value = features[feature_name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise PredictionInputError(f"{feature_name} must be numeric")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise PredictionInputError(f"{feature_name} must be finite")
        validated.append(numeric_value)
    return tuple(validated)


class PredictionService:
    """Validate user features, delegate inference, and emit observability data."""

    def __init__(
        self,
        model: LoadedIrisModel,
        manifest: ModelManifest,
        *,
        logger: PredictionLogger,
        low_confidence_threshold: float,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._model = model
        self._manifest = manifest
        self._logger = logger
        self._low_confidence_threshold = low_confidence_threshold
        self._clock = clock

    def predict(self, features: object) -> PredictionResult:
        """Return a prediction for one exact manifest-shaped feature mapping."""

        try:
            validated_features = validate_features(features, self._manifest)
        except PredictionInputError:
            self._logger.invalid_input()
            raise

        started = self._clock()
        try:
            prediction = self._model.predict([validated_features])
            self._validate_prediction(prediction)
        except Exception as error:
            self._logger.prediction_error(
                valid_input=True,
                error=type(error).__name__,
            )
            raise

        latency_ms = (self._clock() - started) * 1000
        result = PredictionResult(
            predicted_class=prediction.predicted_class,
            confidence=prediction.confidence,
            probabilities=dict(prediction.probabilities),
            latency_ms=latency_ms,
            valid_input=True,
        )
        self._logger.prediction_success(
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            latency_ms=result.latency_ms,
            valid_input=result.valid_input,
        )
        if result.confidence < self._low_confidence_threshold:
            self._logger.metric("LowConfidencePredictionCount", 1.0)
        return result

    def _validate_prediction(self, prediction: IrisPrediction) -> None:
        if not isinstance(prediction, IrisPrediction):
            raise TypeError("model returned an invalid prediction result")
        if prediction.predicted_class not in self._manifest.classes:
            raise ValueError("model returned an unexpected class")
        if tuple(prediction.probabilities) != tuple(self._manifest.classes):
            raise ValueError("model returned an unexpected probability schema")


__all__ = [
    "PredictionInputError",
    "PredictionResult",
    "PredictionService",
    "validate_features",
]
