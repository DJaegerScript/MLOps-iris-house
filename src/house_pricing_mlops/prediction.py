"""Validated online prediction for the House Pricing model."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from house_pricing_mlops.bundle import (
    BundleLoadError,
    LoadedHousePriceModel,
    load_model_bundle_bytes,
)
from house_pricing_mlops.config import HouseSettings
from house_pricing_mlops.schema import (
    MODEL_FEATURES,
    SchemaValidationError,
    validate_single_features,
)


class HousePredictionInputError(ValueError):
    """Stable public error for invalid House online input."""

    def __init__(self, reason: str = "invalid House Pricing input") -> None:
        self.reason = reason
        super().__init__("Invalid House Pricing input")


class HousePredictionLogger(Protocol):
    def invalid_input(self) -> None:
        """Log an invalid input without values."""

    def prediction_success(
        self,
        *,
        predicted_sale_price: float,
        latency_ms: float,
        valid_input: bool,
    ) -> None:
        """Log a successful online prediction."""

    def prediction_error(self, *, valid_input: bool, error: str) -> None:
        """Log a safe prediction error."""


@dataclass(frozen=True, slots=True)
class HousePredictionResult:
    """Public online prediction result."""

    predicted_sale_price: float
    latency_ms: float
    valid_input: bool
    model_version: str
    dataset_version: str


class VersionedObjectStore(Protocol):
    def get_versioned_object(self, bucket: str, key: str, version_id: str) -> bytes:
        """Return exactly one versioned object."""


def load_model_from_s3(
    store: VersionedObjectStore,
    bucket: str,
    key: str,
    version_id: str,
) -> LoadedHousePriceModel:
    """Fetch and verify one exact House model object version."""

    try:
        payload = store.get_versioned_object(bucket, key, version_id)
        return load_model_bundle_bytes(payload)
    except (BundleLoadError, ValueError):
        raise
    except Exception as error:
        raise BundleLoadError("could not retrieve House model artifact") from error


class HousePredictionService:
    """Validate one online request, predict, and emit safe observability data."""

    def __init__(
        self,
        model: Any,
        settings: HouseSettings,
        *,
        logger: HousePredictionLogger,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._model = model
        self._settings = settings
        self._logger = logger
        self._clock = clock

    def predict(self, features: object) -> HousePredictionResult:
        try:
            validated = validate_single_features(features)
        except SchemaValidationError as error:
            self._logger.invalid_input()
            raise HousePredictionInputError(str(error)) from error

        started = self._clock()
        try:
            frame = pd.DataFrame([validated], columns=list(MODEL_FEATURES))
            raw_prediction = np.asarray(
                self._model.predict(frame), dtype=float
            ).reshape(-1)
            if len(raw_prediction) != 1 or not math.isfinite(raw_prediction[0]):
                raise ValueError("model returned an invalid price")
            if raw_prediction[0] < 0:
                raise ValueError("model returned a negative price")
            predicted_sale_price = float(raw_prediction[0])
        except Exception as error:
            self._logger.prediction_error(valid_input=True, error=type(error).__name__)
            raise

        latency_ms = (self._clock() - started) * 1000
        result = HousePredictionResult(
            predicted_sale_price=predicted_sale_price,
            latency_ms=latency_ms,
            valid_input=True,
            model_version=self._settings.model_version,
            dataset_version=self._settings.dataset_version,
        )
        self._logger.prediction_success(
            predicted_sale_price=result.predicted_sale_price,
            latency_ms=result.latency_ms,
            valid_input=result.valid_input,
        )
        return result


__all__ = [
    "HousePredictionInputError",
    "HousePredictionResult",
    "HousePredictionService",
    "load_model_from_s3",
]
