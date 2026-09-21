"""Structured stdout logging for Streamlit and ECS/CloudWatch."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Mapping
from typing import TextIO

from iris_mlops.config import Settings

_METRIC_DEFINITIONS: Mapping[str, str] = {
    "PredictionCount": "Count",
    "PredictionErrorCount": "Count",
    "PredictionLatencyMs": "Milliseconds",
    "LowConfidencePredictionCount": "Count",
    "InvalidInputCount": "Count",
    "ModelLoadFailure": "Count",
}
_DIMENSIONS = ["Environment", "ModelVersion"]


class StructuredLogger:
    """Write one JSON object per line with low-cardinality deployment context."""

    def __init__(
        self,
        settings: Settings,
        *,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._settings = settings
        self._stream = sys.stdout if stream is None else stream
        self._clock = clock

    def event(self, name: str, **fields: object) -> None:
        """Emit a structured application event without request/input payloads."""

        record: dict[str, object] = {
            "event": name,
            "environment": self._settings.environment,
            "model_version": self._settings.model_version,
            "git_commit_sha": self._settings.git_commit_sha,
            "docker_image_version": self._settings.docker_image_version,
            **fields,
        }
        self._write(record)

    def metric(self, name: str, value: float) -> None:
        """Emit a CloudWatch Embedded Metric Format record."""

        try:
            unit = _METRIC_DEFINITIONS[name]
        except KeyError as error:
            raise ValueError(f"unsupported CloudWatch metric: {name}") from error

        record: dict[str, object] = {
            "_aws": {
                "Timestamp": int(self._clock() * 1000),
                "CloudWatchMetrics": [
                    {
                        "Namespace": "IrisMLOps",
                        "Dimensions": [_DIMENSIONS],
                        "Metrics": [{"Name": name, "Unit": unit}],
                    }
                ],
            },
            "Environment": self._settings.environment,
            "ModelVersion": self._settings.model_version,
            name: value,
        }
        self._write(record)

    def startup(self) -> None:
        """Record application startup."""

        self.event("startup")

    def model_load_success(self) -> None:
        """Record a successful model load."""

        self.event("model_load_success")

    def model_load_failure(self, error: str) -> None:
        """Record a model load failure using a caller-provided safe summary."""

        self.event("model_load_failure", error=error)
        self.metric("ModelLoadFailure", 1.0)

    def invalid_input(self) -> None:
        """Record rejected input without serializing its contents."""

        self.event("invalid_input", valid_input=False)
        self.metric("InvalidInputCount", 1.0)

    def prediction_success(
        self,
        *,
        predicted_class: str,
        confidence: float,
        latency_ms: float,
        valid_input: bool,
    ) -> None:
        """Record the required prediction event."""

        self.event(
            "prediction",
            predicted_class=predicted_class,
            confidence=confidence,
            latency_ms=latency_ms,
            valid_input=valid_input,
        )
        self.metric("PredictionCount", 1.0)
        self.metric("PredictionLatencyMs", latency_ms)

    def prediction_error(self, *, valid_input: bool, error: str) -> None:
        """Record an inference failure without serializing input values."""

        self.event("prediction_error", valid_input=valid_input, error=error)
        self.metric("PredictionErrorCount", 1.0)

    def _write(self, record: dict[str, object]) -> None:
        self._stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._stream.flush()


__all__ = ["StructuredLogger"]
