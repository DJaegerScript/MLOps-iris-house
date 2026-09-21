"""Tests for the Streamlit Iris application boundary."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import app
from iris_mlops.manifest import load_manifest
from iris_mlops.prediction import PredictionInputError, PredictionResult

MANIFEST = load_manifest("artifacts/iris-classifier-v1.manifest.json")


@dataclass
class FakeRuntime:
    manifest: object
    service: object


class FakeService:
    def __init__(
        self,
        result: PredictionResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.inputs: list[object] = []

    def predict(self, features: object) -> PredictionResult:
        self.inputs.append(features)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


class FakeStreamlit:
    def __init__(self, *, clicked: bool = True) -> None:
        self.clicked = clicked
        self.calls: list[tuple[str, object]] = []
        self.errors: list[str] = []
        self.values = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }

    def set_page_config(self, **kwargs: object) -> None:
        self.calls.append(("set_page_config", kwargs))

    def title(self, value: str) -> None:
        self.calls.append(("title", value))

    def subheader(self, value: str) -> None:
        self.calls.append(("subheader", value))

    def markdown(self, value: str) -> None:
        self.calls.append(("markdown", value))

    def caption(self, value: str) -> None:
        self.calls.append(("caption", value))

    def number_input(self, label: str, **kwargs: object) -> float:
        key = str(kwargs["key"])
        self.calls.append(("number_input", key))
        return self.values[key]

    def button(self, label: str) -> bool:
        self.calls.append(("button", label))
        return self.clicked

    def success(self, value: str) -> None:
        self.calls.append(("success", value))

    def error(self, value: str) -> None:
        self.errors.append(value)
        self.calls.append(("error", value))

    def write(self, value: object) -> None:
        self.calls.append(("write", value))

    def table(self, value: object) -> None:
        self.calls.append(("table", value))


def _result() -> PredictionResult:
    return PredictionResult(
        predicted_class="setosa",
        confidence=0.98,
        probabilities={
            "setosa": 0.98,
            "versicolor": 0.01,
            "virginica": 0.01,
        },
        latency_ms=12.0,
        valid_input=True,
    )


def test_app_declares_exact_feature_input_keys() -> None:
    assert app.FEATURES == (
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    )


def test_runtime_loader_is_streamlit_resource_cached() -> None:
    assert getattr(app.load_runtime_service, "__wrapped__", None) is not None


def test_startup_renders_inputs_prediction_probabilities_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = FakeStreamlit()
    fake_service = FakeService(result=_result())
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "load_config", lambda: object())
    monkeypatch.setattr(
        app,
        "load_runtime_service",
        lambda _settings, _manifest_path: FakeRuntime(MANIFEST, fake_service),
    )

    app.main()

    input_keys = [value for name, value in fake_st.calls if name == "number_input"]
    assert input_keys == list(app.FEATURES)
    assert ("button", "Predict species") in fake_st.calls
    assert any(
        name == "success" and "setosa" in str(value)
        for name, value in fake_st.calls
    )
    assert any(name == "table" for name, _value in fake_st.calls)
    assert any(MANIFEST.source in str(value) for _name, value in fake_st.calls)
    assert fake_service.inputs == [
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }
    ]


def test_invalid_prediction_input_is_rendered_as_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = FakeStreamlit()
    fake_service = FakeService(
        error=PredictionInputError("petal_width must be finite")
    )
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "load_config", lambda: object())
    monkeypatch.setattr(
        app,
        "load_runtime_service",
        lambda _settings, _manifest_path: FakeRuntime(MANIFEST, fake_service),
    )

    app.main()

    assert any("petal_width must be finite" in message for message in fake_st.errors)


def test_runtime_load_failure_is_rendered_without_aws_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = FakeStreamlit(clicked=False)
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "load_config", lambda: object())
    monkeypatch.setattr(
        app,
        "load_runtime_service",
        lambda _settings, _manifest_path: (_ for _ in ()).throw(
            RuntimeError("model unavailable")
        ),
    )

    app.main()

    assert any("model unavailable" in message for message in fake_st.errors)
