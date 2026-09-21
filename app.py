"""Streamlit product surface for the versioned, pre-trained Iris classifier."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from house_pricing_mlops.bundle import LoadedHousePriceModel
from house_pricing_mlops.config import HouseSettings, load_house_config
from house_pricing_mlops.logging_utils import HouseStructuredLogger
from house_pricing_mlops.prediction import (
    HouseBatchPredictionService,
    HousePredictionInputError,
    HousePredictionService,
)
from house_pricing_mlops.prediction import (
    load_model_from_s3 as load_house_model_from_s3,
)
from house_pricing_mlops.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from iris_mlops.config import Settings, load_config
from iris_mlops.logging_utils import StructuredLogger
from iris_mlops.manifest import ModelManifest, load_manifest
from iris_mlops.model import load_model_from_s3
from iris_mlops.prediction import (
    PredictionInputError,
    PredictionResult,
    PredictionService,
)
from iris_mlops.storage import S3ArtifactStore

FEATURES = (
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
)
IRIS_PRODUCT = "Iris Classifier"
HOUSE_PRODUCT = "House Price Prediction"
MANIFEST_PATH = (
    Path(__file__).resolve().parent / "artifacts" / "iris-classifier-v1.manifest.json"
)
_DEFAULT_VALUES = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}
_HOUSE_FEATURE_LABELS: dict[str, str] = {
    "OverallQual": "Overall Quality",
    "GrLivArea": "Above-Ground Living Area (sq ft)",
    "GarageCars": "Garage Capacity (cars)",
    "TotalBsmtSF": "Total Basement Area",
    "1stFlrSF": "First-Floor Area",
    "YearBuilt": "Year Built",
    "FullBath": "Full Bathrooms",
    "TotRmsAbvGrd": "Total Rooms Above Grade",
    "GarageArea": "Garage Area",
    "Neighborhood": "Neighborhood",
    "KitchenQual": "Kitchen Quality",
    "CentralAir": "Central Air Conditioning",
}
_HOUSE_CATEGORY_LABELS: dict[str, dict[str, str]] = {
    "Neighborhood": {
        "Blmngtn": "Bloomington Heights",
        "Blueste": "Bluestem",
        "BrDale": "Briardale",
        "BrkSide": "Brookside",
        "ClearCr": "Clear Creek",
        "CollgCr": "College Creek",
        "Crawford": "Crawford",
        "Crawfor": "Crawford",
        "Edwards": "Edwards",
        "Gilbert": "Gilbert",
        "IDOTRR": "Iowa DOT and Rail Road",
        "MeadowV": "Meadow Village",
        "Mitchel": "Mitchell",
        "NAmes": "North Ames",
        "NoRidge": "Northridge",
        "NPkVill": "Northpark Villa",
        "NridgHt": "Northridge Heights",
        "NWA": "Northwest Ames",
        "NWAmes": "Northwest Ames",
        "OldTown": "Old Town",
        "Sawyer": "Sawyer",
        "SawyerW": "Sawyer West",
        "Somerst": "Somerset",
        "StoneBr": "Stone Brook",
        "SWISU": "South & West of Iowa State University",
        "Timber": "Timberland",
        "Veenker": "Veenker",
    },
    "KitchenQual": {
        "Ex": "Excellent",
        "Gd": "Good",
        "TA": "Typical/Average",
        "Fa": "Fair",
        "Po": "Poor",
    },
    "CentralAir": {"Y": "Yes", "N": "No"},
}


@dataclass(frozen=True, slots=True)
class AppRuntime:
    """Cached inference objects and metadata required by the page."""

    manifest: ModelManifest
    service: PredictionService


@dataclass(frozen=True, slots=True)
class HouseRuntime:
    """Cached House model, service, and batch prediction objects."""

    model: LoadedHousePriceModel
    service: HousePredictionService
    batch_service: HouseBatchPredictionService


@st.cache_resource(show_spinner=False)
def load_runtime_service(settings: Settings, manifest_path: str) -> AppRuntime:
    """Load the manifest and one immutable S3 model version for the process."""

    manifest = load_manifest(manifest_path)
    logger = StructuredLogger(settings)
    logger.startup()
    try:
        model = load_model_from_s3(
            S3ArtifactStore(),
            settings.model_s3_bucket,
            settings.model_s3_key,
            settings.model_s3_version_id,
            manifest,
        )
    except Exception as error:
        logger.model_load_failure(type(error).__name__)
        raise RuntimeError("model could not be loaded") from error

    logger.model_load_success()
    return AppRuntime(
        manifest=manifest,
        service=PredictionService(
            model,
            manifest,
            logger=logger,
            low_confidence_threshold=settings.low_confidence_threshold,
        ),
    )


@st.cache_resource(show_spinner=False)
def load_house_runtime(settings: HouseSettings) -> HouseRuntime:
    """Load House only after its page is selected and its exact config exists."""

    logger = HouseStructuredLogger(settings)
    logger.startup()
    try:
        model = load_house_model_from_s3(
            S3ArtifactStore(),
            settings.model_s3_bucket,
            settings.model_s3_key,
            settings.model_s3_version_id,
        )
    except Exception as error:
        logger.model_load_failure(type(error).__name__)
        raise RuntimeError("House Pricing model could not be loaded") from error
    logger.model_load_success()
    return HouseRuntime(
        model=model,
        service=HousePredictionService(model, settings, logger=logger),
        batch_service=HouseBatchPredictionService(
            model,
            settings,
            reference_profile=model.reference_profile,
            logger=logger,
        ),
    )


def _render_provenance(manifest: ModelManifest) -> None:
    st.subheader("Model provenance")
    st.caption(
        f"Model version: {manifest.model_version} | "
        f"Framework: {manifest.framework} {manifest.framework_version}"
    )
    st.markdown(
        f"Source: [{manifest.source}]({manifest.source}) "
        f"(immutable revision `{manifest.source_revision}`)"
    )
    metrics: Any = manifest.training_metrics
    cross_validation = metrics["cross_validation"]
    st.caption(
        "Verified source-reported metrics: "
        f"{cross_validation['folds']}-fold CV accuracy "
        f"{cross_validation['accuracy']:.1%}; "
        f"test accuracy {metrics['test_accuracy']:.1%}."
    )


def _render_result(result: PredictionResult) -> None:
    st.success(f"Predicted species: {result.predicted_class}")
    st.write(f"Confidence: {result.confidence:.1%}")
    st.caption(f"Inference latency: {result.latency_ms:.1f} ms")
    st.table(
        [
            {"species": species, "probability": f"{probability:.1%}"}
            for species, probability in result.probabilities.items()
        ]
    )


def _selected_product() -> str:
    sidebar = getattr(st, "sidebar", None)
    if sidebar is None:
        return IRIS_PRODUCT
    return sidebar.radio(
        "Product",
        [IRIS_PRODUCT, HOUSE_PRODUCT],
        index=0,
    )


def _render_iris_page() -> None:
    st.title("Iris species classifier")
    st.markdown(
        "This educational app serves a pre-trained scikit-learn classifier. "
        "Enter four flower measurements to estimate whether the flower is "
        "setosa, versicolor, or virginica."
    )

    try:
        settings = load_config()
        runtime = load_runtime_service(settings, str(MANIFEST_PATH))
    except Exception as error:
        st.error(f"Model unavailable: {error}")
        return

    _render_provenance(runtime.manifest)
    st.subheader("Flower measurements")
    features = {
        feature: st.number_input(
            feature.replace("_", " ").title(),
            min_value=0.0,
            step=0.1,
            value=_DEFAULT_VALUES[feature],
            format="%.2f",
            key=feature,
        )
        for feature in FEATURES
    }

    if st.button("Predict species"):
        try:
            result = runtime.service.predict(features)
        except PredictionInputError as error:
            st.error(f"Validation error: {error.reason}.")
            return
        except Exception:
            st.error("Prediction failed. Please try again.")
            return
        _render_result(result)


def _render_house_page() -> None:
    st.title("House price prediction")
    try:
        settings = load_house_config()
        runtime = load_house_runtime(settings)
    except Exception as error:
        st.error(f"House Pricing unavailable: {error}")
        return

    st.caption(
        f"Model version: {settings.model_version} | "
        f"Dataset version: {settings.dataset_version} | "
        f"Schema: house-price-v1"
    )

    st.subheader("Property prediction")
    features: dict[str, object] = {}
    for feature in NUMERIC_FEATURES:
        features[feature] = st.number_input(
            _HOUSE_FEATURE_LABELS[feature],
            min_value=0.0,
            value=None,
            step=1.0,
            key=feature,
        )
    for feature in CATEGORICAL_FEATURES:
        options = _house_category_options(runtime.model, feature)
        features[feature] = st.selectbox(
            _HOUSE_FEATURE_LABELS[feature],
            options,
            index=None,
            placeholder="Required",
            key=feature,
            format_func=lambda value, feature=feature: _house_category_label(
                feature, value
            ),
        )

    if st.button("Predict house price"):
        try:
            result = runtime.service.predict(features)
        except HousePredictionInputError as error:
            st.error(f"Validation error: {error.reason}.")
        except Exception:
            st.error("House price prediction failed. Please try again.")
        else:
            st.success(f"Predicted sale price: ${result.predicted_sale_price:,.0f}")
            st.caption(f"Inference latency: {result.latency_ms:.1f} ms")

def _house_category_options(model: LoadedHousePriceModel, feature: str) -> list[str]:
    try:
        categories = model.reference_profile["categorical"][feature]["categories"]
    except (AttributeError, KeyError, TypeError):
        return []
    if not isinstance(categories, list):
        return []
    return [str(category) for category in categories]


def _house_category_label(feature: str, value: object) -> str:
    raw_value = str(value)
    explicit_labels = _HOUSE_CATEGORY_LABELS.get(feature, {})
    if raw_value in explicit_labels:
        return explicit_labels[raw_value]

    readable_value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", raw_value)
    readable_value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", readable_value)
    readable_value = re.sub(r"[_-]+", " ", readable_value)
    return " ".join(readable_value.split()).title()


def main() -> None:
    """Render the selected AI product page."""

    st.set_page_config(page_title="Iris Classifier", page_icon="🌸")
    if _selected_product() == HOUSE_PRODUCT:
        _render_house_page()
        return
    _render_iris_page()


if __name__ == "__main__":  # pragma: no cover - Streamlit invokes this module
    main()
