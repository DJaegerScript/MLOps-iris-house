"""Structured stdout and CloudWatch EMF logging for House Pricing."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Mapping
from typing import TextIO

from house_pricing_mlops.config import HouseSettings

_METRIC_DEFINITIONS: Mapping[str, str] = {
    "HousePredictionCount": "Count",
    "HousePredictionErrorCount": "Count",
    "HousePredictionLatencyMs": "Milliseconds",
    "HouseBatchPredictionCount": "Count",
    "HouseBatchRowsProcessed": "Count",
    "HouseInvalidInputCount": "Count",
    "HouseDriftedFeatureCount": "Count",
    "HouseModelLoadFailure": "Count",
    "HouseBatchMAE": "None",
    "HouseBatchRMSE": "None",
}
_DIMENSIONS = ["Environment", "Product", "ModelVersion"]


class HouseStructuredLogger:
    """Emit safe House events and low-cardinality EMF metrics."""

    def __init__(
        self,
        settings: HouseSettings,
        *,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._settings = settings
        self._stream = sys.stdout if stream is None else stream
        self._clock = clock

    def event(self, name: str, **fields: object) -> None:
        record: dict[str, object] = {
            "event": name,
            "product": "house-pricing",
            "environment": self._settings.environment,
            "model_version": self._settings.model_version,
            "dataset_version": self._settings.dataset_version,
            "git_commit_sha": self._settings.git_commit_sha,
            "docker_image_version": self._settings.docker_image_version,
            **fields,
        }
        self._write(record)

    def metric(self, name: str, value: float) -> None:
        try:
            unit = _METRIC_DEFINITIONS[name]
        except KeyError as error:
            raise ValueError(f"unsupported House metric: {name}") from error
        record: dict[str, object] = {
            "_aws": {
                "Timestamp": int(self._clock() * 1000),
                "CloudWatchMetrics": [
                    {
                        "Namespace": "HousePricingMLOps",
                        "Dimensions": [_DIMENSIONS],
                        "Metrics": [{"Name": name, "Unit": unit}],
                    }
                ],
            },
            "Environment": self._settings.environment,
            "Product": "house-pricing",
            "ModelVersion": self._settings.model_version,
            name: value,
        }
        self._write(record)

    def startup(self) -> None:
        self.event("startup")

    def model_load_success(self) -> None:
        self.event("model_load_success")

    def model_load_failure(self, error: str) -> None:
        self.event("model_load_failure", error=error)
        self.metric("HouseModelLoadFailure", 1.0)

    def invalid_input(self) -> None:
        self.event("house_input_validation_failure", valid_input=False)
        self.metric("HouseInvalidInputCount", 1.0)

    def prediction_success(
        self,
        *,
        predicted_sale_price: float,
        latency_ms: float,
        valid_input: bool,
    ) -> None:
        self.event(
            "house_prediction",
            predicted_sale_price=predicted_sale_price,
            latency_ms=latency_ms,
            valid_input=valid_input,
        )
        self.metric("HousePredictionCount", 1.0)
        self.metric("HousePredictionLatencyMs", latency_ms)

    def prediction_error(self, *, valid_input: bool, error: str) -> None:
        self.event("house_prediction_error", valid_input=valid_input, error=error)
        self.metric("HousePredictionErrorCount", 1.0)

    def _write(self, record: dict[str, object]) -> None:
        self._stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._stream.flush()


__all__ = ["HouseStructuredLogger"]
