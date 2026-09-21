"""Validated online prediction for the House Pricing model."""

from __future__ import annotations

import io
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import numpy as np
import pandas as pd

from house_pricing_mlops.bundle import (
    BundleLoadError,
    LoadedHousePriceModel,
    load_model_bundle_bytes,
)
from house_pricing_mlops.config import HouseSettings
from house_pricing_mlops.drift import DriftReport, score_batch
from house_pricing_mlops.schema import (
    MODEL_FEATURES,
    TARGET_NAME,
    SchemaValidationError,
    validate_prediction_frame,
    validate_single_features,
)
from house_pricing_mlops.training import calculate_metrics


class HousePredictionInputError(ValueError):
    """Stable public error for invalid House online input."""

    def __init__(self, reason: str = "invalid House Pricing input") -> None:
        self.reason = reason
        super().__init__("Invalid House Pricing input")


class HouseBatchValidationError(ValueError):
    """Stable public error for unsafe or malformed batch uploads."""


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

    def batch_validation_error(self, *, error: str) -> None:
        """Log a safe batch validation error."""

    def batch_prediction_error(self, *, error: str) -> None:
        """Log a safe batch prediction error."""

    def batch_prediction(
        self,
        *,
        rows_received: int,
        rows_processed: int,
        invalid_rows: int,
        drifted_features: int,
        latency_ms: float,
        batch_mae: float | None,
        batch_rmse: float | None,
    ) -> None:
        """Log a completed batch prediction."""


@dataclass(frozen=True, slots=True)
class HousePredictionResult:
    """Public online prediction result."""

    predicted_sale_price: float
    latency_ms: float
    valid_input: bool
    model_version: str
    dataset_version: str


@dataclass(frozen=True, slots=True)
class HouseBatchPredictionResult:
    """Safe batch output, optional metrics, and drift diagnostics."""

    predictions: pd.DataFrame
    metrics: Mapping[str, float | None] | None
    drift_report: DriftReport
    rows_received: int
    rows_processed: int
    invalid_row_count: int
    model_version: str
    dataset_version: str

    def csv_bytes(self) -> bytes:
        return self.predictions.to_csv(index=False).encode("utf-8")


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


class HouseBatchPredictionService:
    """Validate, predict, evaluate, and score drift for one CSV batch."""

    def __init__(
        self,
        model: Any,
        settings: HouseSettings,
        *,
        reference_profile: Mapping[str, Any],
        logger: HousePredictionLogger,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._model = model
        self._settings = settings
        self._reference_profile = reference_profile
        self._logger = logger
        self._clock = clock

    def predict_csv(
        self,
        upload: object,
        *,
        filename: str,
    ) -> HouseBatchPredictionResult:
        payload = self._read_upload(upload, filename)
        try:
            frame = pd.read_csv(
                io.BytesIO(payload), nrows=self._settings.max_batch_rows + 1
            )
        except pd.errors.EmptyDataError as error:
            self._log_batch_validation("empty CSV file")
            raise HouseBatchValidationError("empty CSV file") from error
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
            self._log_batch_validation("malformed CSV file")
            raise HouseBatchValidationError("malformed CSV file") from error
        if frame.empty:
            self._log_batch_validation("empty CSV file")
            raise HouseBatchValidationError("empty CSV file")
        if len(frame) > self._settings.max_batch_rows:
            self._log_batch_validation("batch exceeds maximum rows")
            raise HouseBatchValidationError("batch exceeds maximum rows")

        allow_labels = TARGET_NAME in frame.columns
        try:
            validation = validate_prediction_frame(frame, allow_labels=allow_labels)
        except SchemaValidationError as error:
            self._log_batch_validation(str(error))
            raise HouseBatchValidationError(str(error)) from error
        if validation.valid_frame.empty:
            self._log_batch_validation("batch contains no valid rows")
            raise HouseBatchValidationError("batch contains no valid rows")

        started = self._clock()
        model_features = validation.valid_frame.loc[:, MODEL_FEATURES]
        try:
            predictions = np.asarray(
                self._model.predict(model_features), dtype=float
            ).reshape(-1)
            if len(predictions) != len(model_features) or not np.isfinite(
                predictions
            ).all():
                raise ValueError("model returned invalid batch predictions")
            if (predictions < 0).any():
                raise ValueError("model returned negative batch predictions")
        except Exception as error:
            self._log_batch_prediction_error(type(error).__name__)
            raise

        drift_report = score_batch(
            self._reference_profile,
            model_features,
            invalid_row_count=validation.invalid_row_count,
        )
        labels = validation.valid_frame[TARGET_NAME] if allow_labels else None
        metrics = calculate_metrics(labels, predictions) if labels is not None else None
        timestamp = datetime.now(UTC).isoformat()
        output = pd.DataFrame(index=validation.valid_frame.index)
        if "Id" in validation.valid_frame:
            output["Id"] = validation.valid_frame["Id"].to_numpy()
        if allow_labels:
            output[TARGET_NAME] = labels.to_numpy()
        output["predicted_sale_price"] = predictions
        output["model_version"] = self._settings.model_version
        output["prediction_timestamp"] = timestamp
        output = output.reset_index(drop=True)
        latency_ms = (self._clock() - started) * 1000
        self._logger.batch_prediction(
            rows_received=len(frame),
            rows_processed=len(output),
            invalid_rows=validation.invalid_row_count,
            drifted_features=drift_report.number_drifted_features,
            latency_ms=latency_ms,
            batch_mae=metrics.get("mae") if metrics else None,
            batch_rmse=metrics.get("rmse") if metrics else None,
        )
        return HouseBatchPredictionResult(
            predictions=output,
            metrics=metrics,
            drift_report=drift_report,
            rows_received=len(frame),
            rows_processed=len(output),
            invalid_row_count=validation.invalid_row_count,
            model_version=self._settings.model_version,
            dataset_version=self._settings.dataset_version,
        )

    def _read_upload(self, upload: object, filename: str) -> bytes:
        if not isinstance(filename, str) or not filename.lower().endswith(".csv"):
            self._log_batch_validation("CSV file is required")
            raise HouseBatchValidationError("CSV file is required")
        if isinstance(upload, bytes):
            payload = upload
        elif isinstance(upload, bytearray):
            payload = bytes(upload)
        else:
            getvalue = getattr(upload, "getvalue", None)
            if callable(getvalue):
                payload = getvalue()
            else:
                read = getattr(upload, "read", None)
                payload = (
                    read(self._settings.max_upload_bytes + 1)
                    if callable(read)
                    else None
                )
        if not isinstance(payload, bytes):
            self._log_batch_validation("uploaded file could not be read")
            raise HouseBatchValidationError("uploaded file could not be read")
        if len(payload) > self._settings.max_upload_bytes:
            self._log_batch_validation("uploaded file exceeds maximum size")
            raise HouseBatchValidationError("uploaded file exceeds maximum size")
        return payload

    def _log_batch_validation(self, error: str) -> None:
        self._logger.batch_validation_error(error=error)

    def _log_batch_prediction_error(self, error: str) -> None:
        self._logger.batch_prediction_error(error=error)


__all__ = [
    "HouseBatchPredictionResult",
    "HouseBatchPredictionService",
    "HouseBatchValidationError",
    "HousePredictionInputError",
    "HousePredictionResult",
    "HousePredictionService",
    "load_model_from_s3",
]
