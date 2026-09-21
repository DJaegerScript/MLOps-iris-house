"""Streamlit product surface for the versioned, pre-trained Iris classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

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


def main() -> None:
    """Render the Iris classifier page and handle one user prediction."""

    st.set_page_config(page_title="Iris Classifier", page_icon="🌸")
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


if __name__ == "__main__":  # pragma: no cover - Streamlit invokes this module
    main()
