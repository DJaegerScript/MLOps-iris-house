"""Tests for the House Pricing Streamlit page boundary."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

import app
from house_pricing_mlops.prediction import HousePredictionResult
from house_pricing_mlops.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class FakeSidebar:
    def __init__(self, selected: str) -> None:
        self.selected = selected
        self.calls: list[tuple[str, object]] = []

    def radio(self, label: str, options: object, **kwargs: object) -> str:
        self.calls.append((label, options))
        return self.selected


class FakeHouseStreamlit:
    def __init__(self, *, selected: str = "House Price Prediction") -> None:
        self.sidebar = FakeSidebar(selected)
        self.calls: list[tuple[str, object]] = []
        self.errors: list[str] = []
        self.input_labels: dict[str, str] = {}
        self.selectbox_options: dict[str, list[str]] = {}
        self.selectbox_formatters: dict[str, object] = {}
        self.formatted_options: dict[str, list[str]] = {}
        self.values = {feature: 1.0 for feature in NUMERIC_FEATURES}
        self.values.update({feature: "known" for feature in CATEGORICAL_FEATURES})
        self.values.update(
            {
                "Neighborhood": "CollgCr",
                "KitchenQual": "TA",
                "CentralAir": "Y",
            }
        )

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
        self.input_labels[key] = label
        self.calls.append(("number_input", key))
        return float(self.values[key])

    def selectbox(self, label: str, options: object, **kwargs: object) -> str:
        key = str(kwargs["key"])
        option_values = [str(option) for option in options]
        format_func = kwargs.get("format_func")
        self.input_labels[key] = label
        self.selectbox_options[key] = option_values
        self.selectbox_formatters[key] = format_func
        self.formatted_options[key] = [
            str(format_func(option)) if callable(format_func) else option
            for option in option_values
        ]
        self.calls.append(("selectbox", key))
        return str(self.values[key])

    def button(self, label: str, **kwargs: object) -> bool:
        self.calls.append(("button", label))
        return label == "Predict house price"

    def success(self, value: str) -> None:
        self.calls.append(("success", value))

    def error(self, value: str) -> None:
        self.errors.append(value)
        self.calls.append(("error", value))

    def warning(self, value: str) -> None:
        self.calls.append(("warning", value))

    def info(self, value: str) -> None:
        self.calls.append(("info", value))

    def write(self, value: object) -> None:
        self.calls.append(("write", value))

    def table(self, value: object) -> None:
        self.calls.append(("table", value))

    def download_button(self, label: str, **kwargs: object) -> None:
        self.calls.append(("download_button", label))

    def file_uploader(self, label: str, **kwargs: object) -> None:
        self.calls.append(("file_uploader", label))
        return None


@dataclass
class FakeHouseService:
    result: HousePredictionResult | None = None
    error: Exception | None = None
    inputs: list[object] | None = None

    def predict(self, features: object) -> HousePredictionResult:
        if self.inputs is None:
            self.inputs = []
        self.inputs.append(features)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


@dataclass
class FakeHouseRuntime:
    service: FakeHouseService
    batch_service: object = field(default_factory=SimpleNamespace)
    model: object = field(
        default_factory=lambda: SimpleNamespace(
            metrics={},
            reference_profile={"categorical": {}},
        )
    )


def _result() -> HousePredictionResult:
    return HousePredictionResult(
        predicted_sale_price=215000.0,
        latency_ms=18.4,
        valid_input=True,
        model_version="v1",
        dataset_version="v1",
    )


def test_house_page_renders_all_declared_inputs_versions_and_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = FakeHouseStreamlit()
    service = FakeHouseService(result=_result())
    settings = SimpleNamespace(model_version="v1", dataset_version="v1")
    house_model = SimpleNamespace(
        metrics={
            "test_metrics": {
                "mae": 17098.44,
                "rmse": 27672.43,
                "r2": 0.8994,
                "rmsle": 0.1352,
            }
        },
        reference_profile={
            "categorical": {
                "Neighborhood": {"categories": ["CollgCr"]},
                "KitchenQual": {"categories": ["TA"]},
                "CentralAir": {"categories": ["Y"]},
            }
        },
    )
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "load_house_config", lambda: settings)
    monkeypatch.setattr(
        app,
        "load_house_runtime",
        lambda _settings: FakeHouseRuntime(service, model=house_model),
    )

    app.main()

    input_keys = [
        value
        for name, value in fake_st.calls
        if name in {"number_input", "selectbox"}
    ]
    assert input_keys == [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    assert fake_st.input_labels == {
        "OverallQual": "Overall Quality",
        "GrLivArea": "Above-Ground Living Area (sq ft)",
        "GarageCars": "Garage Capacity (cars)",
        "TotalBsmtSF": "Total Basement Area (sq ft)",
        "1stFlrSF": "First-Floor Area (sq ft)",
        "YearBuilt": "Year Built",
        "FullBath": "Full Bathrooms",
        "TotRmsAbvGrd": "Total Rooms Above Grade",
        "GarageArea": "Garage Area (sq ft)",
        "Neighborhood": "Neighborhood",
        "KitchenQual": "Kitchen Quality",
        "CentralAir": "Central Air Conditioning",
    }
    assert fake_st.selectbox_options == {
        "Neighborhood": ["CollgCr"],
        "KitchenQual": ["TA"],
        "CentralAir": ["Y"],
    }
    assert fake_st.formatted_options == {
        "Neighborhood": ["College Creek"],
        "KitchenQual": ["Typical/Average"],
        "CentralAir": ["Yes"],
    }
    assert not any(
        name == "subheader" and value == "Batch prediction"
        for name, value in fake_st.calls
    )
    assert not any(name == "file_uploader" for name, _ in fake_st.calls)
    assert any(
        name == "success" and "215,000" in str(value)
        for name, value in fake_st.calls
    )
    assert any(
        "House Price Prediction" in str(value)
        for _, value in fake_st.sidebar.calls
    )
    assert not any(
        "This educational model learns from selected Ames, Iowa features in the "
        "Kaggle House Prices dataset. It demonstrates validation, preprocessing, "
        "training provenance, and versioned production serving."
        in str(value)
        for name, value in fake_st.calls
        if name == "markdown"
    )
    assert not any(
        name == "info" and "historical Ames" in str(value)
        for name, value in fake_st.calls
    )
    assert not any(
        name == "write"
        and any(
            label in str(value)
            for label in ("Test MAE", "Test RMSE", "Test R²", "Test RMSLE")
        )
        for name, value in fake_st.calls
    )
    assert service.inputs is not None
    assert set(service.inputs[0]) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert service.inputs[0]["Neighborhood"] == "CollgCr"
    assert service.inputs[0]["KitchenQual"] == "TA"
    assert service.inputs[0]["CentralAir"] == "Y"


def test_house_load_failure_is_rendered_as_house_specific_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = FakeHouseStreamlit()
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "load_house_config", lambda: object())
    monkeypatch.setattr(
        app,
        "load_house_runtime",
        lambda _settings: (_ for _ in ()).throw(
            RuntimeError("House model unavailable")
        ),
    )

    app.main()

    assert any("House Pricing unavailable" in message for message in fake_st.errors)


def test_iris_selection_does_not_load_house_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = FakeHouseStreamlit(selected="Iris Classifier")
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(
        app,
        "load_house_config",
        lambda: (_ for _ in ()).throw(AssertionError("House config was loaded")),
    )
    monkeypatch.setattr(app, "load_config", lambda: object())
    monkeypatch.setattr(
        app,
        "load_runtime_service",
        lambda _settings, _manifest_path: (_ for _ in ()).throw(
            RuntimeError("Iris model unavailable")
        ),
    )

    app.main()

    assert any("Iris model unavailable" in message for message in fake_st.errors)


def test_house_category_label_uses_readable_fallback() -> None:
    assert app._house_category_label("Neighborhood", "NorthWest_Corner") == (
        "North West Corner"
    )


def test_house_runtime_variant_skips_product_selector_and_iris_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = FakeHouseStreamlit()
    rendered: list[str] = []
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setenv("APP_VARIANT", "house")
    monkeypatch.setattr(app, "_render_house_page", lambda: rendered.append("house"))
    monkeypatch.setattr(
        app,
        "_selected_product",
        lambda: pytest.fail("House runtime must not render product navigation"),
    )

    app.main()

    assert rendered == ["house"]
    assert fake_st.calls == [
        (
            "set_page_config",
            {"page_title": "House Price Prediction", "page_icon": "🏠"},
        )
    ]


def test_unsupported_runtime_variant_fails_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = FakeHouseStreamlit()
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setenv("APP_VARIANT", "unknown")
    monkeypatch.setattr(
        app,
        "_selected_product",
        lambda: pytest.fail("Unsupported runtime must not render product navigation"),
    )

    app.main()

    assert any(
        "Unsupported application variant" in message for message in fake_st.errors
    )
