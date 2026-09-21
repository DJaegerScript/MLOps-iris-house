"""Streamlit product surface for the versioned, pre-trained Iris classifier."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from house_pricing_mlops.bundle import LoadedHousePriceModel
from house_pricing_mlops.config import HouseSettings, load_house_config
from house_pricing_mlops.logging_utils import HouseStructuredLogger
from house_pricing_mlops.prediction import (
    HouseBatchPredictionService,
    HouseBatchValidationError,
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
    st.info(
        "Educational model warning: this model was trained on historical Ames, "
        "Iowa data and should not be treated as a current appraisal or a model "
        "for other geographies."
    )
    _render_house_metrics(runtime.model.metrics)

    st.subheader("Single-property prediction")
    features: dict[str, object] = {}
    for feature in NUMERIC_FEATURES:
        features[feature] = st.number_input(
            feature,
            min_value=0.0,
            value=None,
            step=1.0,
            key=feature,
        )
    for feature in CATEGORICAL_FEATURES:
        options = _house_category_options(runtime.model, feature)
        features[feature] = st.selectbox(
            feature,
            options,
            index=None,
            placeholder="Required",
            key=feature,
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

    uploader = getattr(st, "file_uploader", None)
    if not callable(uploader):
        return
    st.subheader("Batch prediction")
    uploaded_file = uploader("Upload a CSV of properties", type=["csv"])
    if uploaded_file is None or not st.button("Run batch prediction"):
        return
    try:
        result = runtime.batch_service.predict_csv(
            uploaded_file,
            filename=getattr(uploaded_file, "name", "uploaded.csv"),
        )
    except HouseBatchValidationError as error:
        st.error(f"Batch validation error: {error}")
        return
    except Exception:
        st.error("Batch prediction failed. Please try again.")
        return
    st.write(result.predictions.head(10))
    st.download_button(
        "Download predictions",
        data=result.csv_bytes(),
        file_name="house_predictions.csv",
        mime="text/csv",
    )
    if result.metrics is not None:
        st.write({name.upper(): value for name, value in result.metrics.items()})
    st.write("Per-feature drift scores", result.drift_report.per_feature_scores)
    st.write("Unknown-category values", result.drift_report.unknown_category_count)
    st.caption(
        f"Rows processed: {result.rows_processed}; "
        f"invalid rows: {result.invalid_row_count}; "
        f"drifted features: {result.drift_report.number_drifted_features}."
    )
    if result.drift_report.sample_size_warning:
        st.warning(
            "This batch is smaller than the educational drift-monitoring "
            "sample-size guideline; one-row predictions cannot reliably measure "
            "input drift."
        )


def _house_category_options(model: LoadedHousePriceModel, feature: str) -> list[str]:
    try:
        categories = model.reference_profile["categorical"][feature]["categories"]
    except (AttributeError, KeyError, TypeError):
        categories = []
    return [str(category) for category in categories] or ["Known category"]


def _render_house_metrics(metrics: Mapping[str, object]) -> None:
    test_metrics = metrics.get("test_metrics") if isinstance(metrics, Mapping) else None
    if not isinstance(test_metrics, Mapping):
        return
    st.write(
        {
            "Test MAE": test_metrics.get("mae"),
            "Test RMSE": test_metrics.get("rmse"),
            "Test R²": test_metrics.get("r2"),
            "Test RMSLE": test_metrics.get("rmsle"),
        }
    )


def main() -> None:
    """Render the selected AI product page."""

    st.set_page_config(page_title="Iris Classifier", page_icon="🌸")
    if _selected_product() == HOUSE_PRODUCT:
        _render_house_page()
        return
    _render_iris_page()


if __name__ == "__main__":  # pragma: no cover - Streamlit invokes this module
    main()
