"""Tests for secure House Pricing batch prediction."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from house_pricing_mlops.config import load_house_config
from house_pricing_mlops.drift import build_reference_profile
from house_pricing_mlops.prediction import (
    HouseBatchPredictionService,
    HouseBatchValidationError,
)
from house_pricing_mlops.schema import MODEL_FEATURES

FIXTURE_PATH = Path("tests/fixtures/house_prices/train.csv")


def _settings(**overrides: str) -> object:
    values = {
        "HOUSE_MODEL_S3_BUCKET": "bucket",
        "HOUSE_MODEL_S3_KEY": "models/house-price/v1/model.tar.gz",
        "HOUSE_MODEL_S3_VERSION_ID": "version-123",
        "HOUSE_MODEL_VERSION": "v1",
        "HOUSE_DATASET_VERSION": "v1",
        "HOUSE_ENVIRONMENT": "test",
        "GIT_COMMIT_SHA": "abc123",
        "DOCKER_IMAGE_VERSION": "image-abc123",
    }
    values.update(overrides)
    return load_house_config(values)


class FakeModel:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features["GrLivArea"].astype(float).to_numpy() * 100.0


class FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []
        self.metrics: list[tuple[str, float]] = []

    def event(self, name: str, **fields: object) -> None:
        self.events.append((name, fields))

    def metric(self, name: str, value: float) -> None:
        self.metrics.append((name, value))

    def batch_prediction(self, **fields: object) -> None:
        self.event("house_batch_prediction", **fields)

    def batch_validation_error(self, *, error: str) -> None:
        self.event("house_batch_validation_error", error=error)

    def prediction_error(self, *, valid_input: bool, error: str) -> None:
        self.event("house_prediction_error", valid_input=valid_input, error=error)


def _service(
    settings=None, logger=None
) -> tuple[HouseBatchPredictionService, FakeLogger]:
    frame = pd.read_csv(FIXTURE_PATH)
    event_logger = logger or FakeLogger()
    service = HouseBatchPredictionService(
        FakeModel(),
        settings or _settings(),
        reference_profile=build_reference_profile(frame.loc[:, MODEL_FEATURES]),
        logger=event_logger,
    )
    return service, event_logger


def test_labeled_batch_returns_predictions_metrics_and_required_columns() -> None:
    frame = pd.read_csv(FIXTURE_PATH).iloc[:3]
    service, logger = _service()

    result = service.predict_csv(
        io.BytesIO(frame.to_csv(index=False).encode()), filename="houses.csv"
    )

    assert list(result.predictions.columns) == [
        "Id",
        "SalePrice",
        "predicted_sale_price",
        "model_version",
        "prediction_timestamp",
    ]
    assert len(result.predictions) == 3
    assert set(result.metrics) == {"mae", "rmse", "r2", "rmsle"}
    assert result.rows_received == 3
    assert result.rows_processed == 3
    assert result.invalid_row_count == 0
    assert logger.events[-1][0] == "house_batch_prediction"


def test_unlabeled_batch_returns_predictions_without_accuracy_claims() -> None:
    frame = pd.read_csv(FIXTURE_PATH).drop(columns=["SalePrice"]).iloc[:2]
    service, _logger = _service()

    result = service.predict_csv(
        io.BytesIO(frame.to_csv(index=False).encode()), filename="houses.csv"
    )

    assert result.metrics is None
    assert "SalePrice" not in result.predictions.columns
    assert "predicted_sale_price" in result.predictions


def test_invalid_rows_are_reported_while_valid_rows_are_processed() -> None:
    frame = pd.read_csv(FIXTURE_PATH).iloc[:3].copy()
    frame["GrLivArea"] = frame["GrLivArea"].astype(object)
    frame.loc[1, "GrLivArea"] = "not-a-number"
    frame.loc[2, "Neighborhood"] = "NewNeighborhood"
    service, _logger = _service()

    result = service.predict_csv(
        io.BytesIO(frame.to_csv(index=False).encode()), filename="houses.csv"
    )

    assert result.rows_received == 3
    assert result.rows_processed == 2
    assert result.invalid_row_count == 1
    assert result.drift_report.unknown_category_count == 1


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("houses.txt", b"not csv", "CSV"),
        ("houses.csv", b"OverallQual,SalePrice\n7,200000\n", "required"),
        ("houses.csv", b"", "empty"),
    ],
)
def test_batch_validation_errors_are_clear(
    filename: str, content: bytes, message: str
) -> None:
    service, _logger = _service()

    with pytest.raises(HouseBatchValidationError, match=message):
        service.predict_csv(io.BytesIO(content), filename=filename)


def test_batch_rejects_unexpected_columns() -> None:
    frame = pd.read_csv(FIXTURE_PATH).iloc[:1].assign(Address="private")
    service, _logger = _service()

    with pytest.raises(HouseBatchValidationError, match="unexpected"):
        service.predict_csv(
            io.BytesIO(frame.to_csv(index=False).encode()), filename="houses.csv"
        )


def test_batch_rejects_size_and_row_limits() -> None:
    frame = pd.read_csv(FIXTURE_PATH)
    service, _logger = _service(
        settings=_settings(HOUSE_MAX_UPLOAD_BYTES="10", HOUSE_MAX_BATCH_ROWS="2")
    )

    with pytest.raises(HouseBatchValidationError, match="size"):
        service.predict_csv(
            io.BytesIO(frame.iloc[:1].to_csv(index=False).encode()),
            filename="houses.csv",
        )

    service, _logger = _service(settings=_settings(HOUSE_MAX_BATCH_ROWS="2"))
    with pytest.raises(HouseBatchValidationError, match="rows"):
        service.predict_csv(
            io.BytesIO(frame.to_csv(index=False).encode()), filename="houses.csv"
        )
