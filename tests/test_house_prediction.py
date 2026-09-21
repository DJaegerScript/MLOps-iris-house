"""Tests for House Pricing online prediction and versioned model loading."""

from pathlib import Path

import numpy as np
import pytest

from house_pricing_mlops.config import load_house_config
from house_pricing_mlops.prediction import (
    HousePredictionInputError,
    HousePredictionService,
    load_model_from_s3,
)
from house_pricing_mlops.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES

FIXTURE_PATH = Path("tests/fixtures/house_prices/train.csv")


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


def _features() -> dict[str, object]:
    values: dict[str, object] = {feature: 1.0 for feature in NUMERIC_FEATURES}
    values.update({feature: "known" for feature in CATEGORICAL_FEATURES})
    return values


class FakeModel:
    def __init__(self, value: float = 215000.0) -> None:
        self.value = value
        self.calls: list[object] = []

    def predict(self, features: object) -> np.ndarray:
        self.calls.append(features)
        return np.asarray([self.value])


class FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []
        self.metrics: list[tuple[str, float]] = []

    def event(self, name: str, **fields: object) -> None:
        self.events.append((name, fields))

    def metric(self, name: str, value: float) -> None:
        self.metrics.append((name, value))

    def invalid_input(self) -> None:
        self.event("house_input_validation_failure", valid_input=False)

    def prediction_success(
        self, *, predicted_sale_price: float, latency_ms: float, valid_input: bool
    ) -> None:
        self.event(
            "house_prediction",
            predicted_sale_price=predicted_sale_price,
            latency_ms=latency_ms,
            valid_input=valid_input,
        )

    def prediction_error(self, *, valid_input: bool, error: str) -> None:
        self.event("house_prediction_error", valid_input=valid_input, error=error)


def test_single_prediction_returns_price_versions_and_latency() -> None:
    logger = FakeLogger()
    service = HousePredictionService(
        FakeModel(), _settings(), logger=logger, clock=iter([10.0, 10.018]).__next__
    )

    result = service.predict(_features())

    assert result.predicted_sale_price == 215000.0
    assert result.model_version == "v1"
    assert result.dataset_version == "v1"
    assert result.latency_ms == pytest.approx(18.0)
    assert result.valid_input is True
    assert logger.events[-1][0] == "house_prediction"


@pytest.mark.parametrize(
    "invalid",
    [
        {},
        {**_features(), "GrLivArea": "large"},
        {**_features(), "Neighborhood": ""},
        {**_features(), "YearBuilt": float("nan")},
    ],
)
def test_invalid_online_input_is_rejected_before_model_call(
    invalid: dict[str, object],
) -> None:
    model = FakeModel()
    logger = FakeLogger()
    service = HousePredictionService(model, _settings(), logger=logger)

    with pytest.raises(HousePredictionInputError):
        service.predict(invalid)

    assert model.calls == []
    assert logger.events[-1][0] == "house_input_validation_failure"


def test_versioned_s3_loader_uses_exact_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Store:
        def __init__(self) -> None:
            self.arguments: tuple[str, str, str] | None = None

        def get_versioned_object(self, bucket: str, key: str, version_id: str) -> bytes:
            self.arguments = (bucket, key, version_id)
            return b"bundle"

    store = Store()
    monkeypatch.setattr(
        "house_pricing_mlops.prediction.load_model_bundle_bytes",
        lambda payload: payload,
    )

    loaded = load_model_from_s3(
        store,
        "bucket",
        "models/house-price/v1/model.tar.gz",
        "version-123",
    )

    assert loaded == b"bundle"
    assert store.arguments == (
        "bucket",
        "models/house-price/v1/model.tar.gz",
        "version-123",
    )
