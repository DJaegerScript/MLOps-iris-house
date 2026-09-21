"""Tests for House Pricing JSON and CloudWatch EMF logging."""

import io
import json

from house_pricing_mlops.config import load_house_config
from house_pricing_mlops.logging_utils import HouseStructuredLogger


def _settings():
    return load_house_config(
        {
            "HOUSE_MODEL_S3_BUCKET": "bucket",
            "HOUSE_MODEL_S3_KEY": "models/house-price/v1/model.tar.gz",
            "HOUSE_MODEL_S3_VERSION_ID": "version-123",
            "HOUSE_MODEL_VERSION": "v1",
            "HOUSE_DATASET_VERSION": "v1",
            "HOUSE_ENVIRONMENT": "test",
            "GIT_COMMIT_SHA": "abc123",
            "DOCKER_IMAGE_VERSION": "image-abc123",
        }
    )


def _records(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_house_prediction_event_has_required_metadata_without_inputs() -> None:
    stream = io.StringIO()
    logger = HouseStructuredLogger(
        _settings(), stream=stream, clock=lambda: 1700000000.0
    )

    logger.prediction_success(
        predicted_sale_price=215000.0,
        latency_ms=18.4,
        valid_input=True,
    )

    record = _records(stream)[0]
    assert record["event"] == "house_prediction"
    assert record["product"] == "house-pricing"
    assert record["model_version"] == "v1"
    assert record["dataset_version"] == "v1"
    assert record["predicted_sale_price"] == 215000.0
    assert record["latency_ms"] == 18.4
    assert record["valid_input"] is True
    assert "GrLivArea" not in record
    assert record["git_commit_sha"] == "abc123"
    assert record["docker_image_version"] == "image-abc123"


def test_house_emf_uses_only_low_cardinality_dimensions_and_all_metrics() -> None:
    stream = io.StringIO()
    logger = HouseStructuredLogger(
        _settings(), stream=stream, clock=lambda: 1700000000.0
    )

    logger.metric("HousePredictionCount", 1.0)
    logger.metric("HouseBatchMAE", 123.0)

    records = _records(stream)
    for record in records:
        emf = record["_aws"]["CloudWatchMetrics"][0]
        assert emf["Dimensions"] == [["Environment", "Product", "ModelVersion"]]
    assert records[0]["HousePredictionCount"] == 1.0
    assert records[1]["HouseBatchMAE"] == 123.0
