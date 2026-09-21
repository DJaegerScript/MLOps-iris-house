"""Tests for stdout JSON logging and CloudWatch EMF records."""

from __future__ import annotations

import io
import json

from iris_mlops.config import Settings
from iris_mlops.logging_utils import StructuredLogger


def _settings() -> Settings:
    return Settings(
        model_s3_bucket="iris-mlops-models",
        model_s3_key="iris/v1/model.tar.gz",
        model_s3_version_id="version-id",
        model_version="v1",
        environment="test",
        git_commit_sha="abc123",
        docker_image_version="image-abc123",
        low_confidence_threshold=0.7,
    )


def _records(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_startup_and_model_load_events_include_deployment_metadata() -> None:
    stream = io.StringIO()
    logger = StructuredLogger(_settings(), stream=stream, clock=lambda: 1700000000.0)

    logger.startup()
    logger.model_load_success()
    logger.model_load_failure("load failed")

    records = [record for record in _records(stream) if "event" in record]
    assert [record["event"] for record in records] == [
        "startup",
        "model_load_success",
        "model_load_failure",
    ]
    for record in records:
        assert record["git_commit_sha"] == "abc123"
        assert record["docker_image_version"] == "image-abc123"
        assert record["model_version"] == "v1"
    assert records[-1]["error"] == "load failed"


def test_prediction_success_event_has_required_fields_and_no_input_values() -> None:
    stream = io.StringIO()
    logger = StructuredLogger(_settings(), stream=stream, clock=lambda: 1700000000.0)

    logger.prediction_success(
        predicted_class="setosa",
        confidence=0.98,
        latency_ms=12.0,
        valid_input=True,
    )

    record = _records(stream)[0]
    assert record["event"] == "prediction"
    assert record["model_version"] == "v1"
    assert record["predicted_class"] == "setosa"
    assert record["confidence"] == 0.98
    assert record["latency_ms"] == 12.0
    assert record["valid_input"] is True
    assert "sepal_length" not in record


def test_metric_event_uses_only_environment_and_model_version_dimensions() -> None:
    stream = io.StringIO()
    logger = StructuredLogger(_settings(), stream=stream, clock=lambda: 1700000000.0)

    logger.metric("PredictionCount", 1.0)

    record = _records(stream)[0]
    assert record["Environment"] == "test"
    assert record["ModelVersion"] == "v1"
    emf = record["_aws"]
    assert emf["CloudWatchMetrics"][0]["Dimensions"] == [[
        "Environment",
        "ModelVersion",
    ]]
    assert emf["CloudWatchMetrics"][0]["Metrics"] == [
        {"Name": "PredictionCount", "Unit": "Count"}
    ]
    assert record["PredictionCount"] == 1.0
