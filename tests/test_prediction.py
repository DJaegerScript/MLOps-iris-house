"""Tests for Iris feature validation and prediction service behavior."""

from __future__ import annotations

import math

import pytest

from iris_mlops.manifest import load_manifest
from iris_mlops.model import IrisPrediction
from iris_mlops.prediction import PredictionInputError, PredictionService

MANIFEST = load_manifest("artifacts/iris-classifier-v1.manifest.json")


class FakeLoadedModel:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def predict(self, features: object) -> IrisPrediction:
        self.calls.append(features)
        return IrisPrediction(
            predicted_class="setosa",
            confidence=0.98,
            probabilities={
                "setosa": 0.98,
                "versicolor": 0.01,
                "virginica": 0.01,
            },
        )


class FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []
        self.metrics: list[tuple[str, float, dict[str, object]]] = []

    def event(self, name: str, **fields: object) -> None:
        self.events.append((name, fields))

    def metric(self, name: str, value: float, **fields: object) -> None:
        self.metrics.append((name, value, fields))

    def invalid_input(self) -> None:
        self.event("invalid_input", valid_input=False)

    def prediction_success(
        self,
        *,
        predicted_class: str,
        confidence: float,
        latency_ms: float,
        valid_input: bool,
    ) -> None:
        self.event(
            "prediction",
            predicted_class=predicted_class,
            confidence=confidence,
            latency_ms=latency_ms,
            valid_input=valid_input,
        )

    def prediction_error(self, *, valid_input: bool, error: str) -> None:
        self.event("prediction_error", valid_input=valid_input, error=error)


def _features() -> dict[str, float]:
    return {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }


def _service(
    model: FakeLoadedModel | None = None,
    logger: FakeLogger | None = None,
    clock_values: list[float] | None = None,
) -> tuple[PredictionService, FakeLoadedModel, FakeLogger]:
    loaded_model = model or FakeLoadedModel()
    event_logger = logger or FakeLogger()
    values = iter(clock_values or [10.0, 10.012])
    return (
        PredictionService(
            loaded_model,
            MANIFEST,
            logger=event_logger,
            low_confidence_threshold=0.7,
            clock=lambda: next(values),
        ),
        loaded_model,
        event_logger,
    )


def test_valid_prediction_returns_class_probabilities_and_latency() -> None:
    service, model, _logger = _service()

    result = service.predict(_features())

    assert result.predicted_class == "setosa"
    assert result.confidence == pytest.approx(0.98)
    assert result.probabilities == {
        "setosa": pytest.approx(0.98),
        "versicolor": pytest.approx(0.01),
        "virginica": pytest.approx(0.01),
    }
    assert result.latency_ms == pytest.approx(12.0)
    assert result.valid_input is True
    assert model.calls == [[tuple(_features().values())]]


@pytest.mark.parametrize(
    "invalid_features",
    [
        {},
        {**_features(), "extra": 1.0},
        {key: value for key, value in _features().items() if key != "petal_width"},
        {**_features(), "petal_width": "0.2"},
        {**_features(), "petal_width": math.nan},
        {**_features(), "petal_width": math.inf},
        {**_features(), "petal_width": True},
    ],
)
def test_invalid_input_is_rejected_before_model_call(
    invalid_features: dict[str, object],
) -> None:
    service, model, logger = _service()

    with pytest.raises(PredictionInputError, match="^Invalid prediction input$"):
        service.predict(invalid_features)

    assert model.calls == []
    assert logger.events[-1][0] == "invalid_input"
    assert logger.events[-1][1]["valid_input"] is False


def test_prediction_error_is_logged_without_raw_input() -> None:
    model = FakeLoadedModel()
    model.predict = lambda _features: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[method-assign]
    service, _model, logger = _service(model=model)

    with pytest.raises(RuntimeError, match="boom"):
        service.predict(_features())

    name, fields = logger.events[-1]
    assert name == "prediction_error"
    assert fields["valid_input"] is True
    assert "sepal_length" not in str(fields)
    assert "5.1" not in str(fields)


def test_low_confidence_prediction_emits_low_confidence_metric() -> None:
    model = FakeLoadedModel()
    model.predict = lambda _features: IrisPrediction(  # type: ignore[method-assign]
        predicted_class="setosa",
        confidence=0.5,
        probabilities={"setosa": 0.5, "versicolor": 0.3, "virginica": 0.2},
    )
    service, _model, logger = _service(model=model)

    service.predict(_features())

    assert ("LowConfidencePredictionCount", 1.0, {}) in logger.metrics
