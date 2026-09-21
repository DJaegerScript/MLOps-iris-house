"""Tests for validated Iris artifact schemas and inference contracts."""

from __future__ import annotations

import hashlib
import io
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from iris_mlops.manifest import ModelManifest, load_manifest
from iris_mlops.model import (
    ArtifactSchemaError,
    LoadedIrisModel,
    load_verified_model,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "artifacts" / "iris-classifier-v1.manifest.json"
ARTIFACT_NAMES = (
    "iris_model.pkl",
    "scaler.pkl",
    "label_encoder.pkl",
    "metadata.pkl",
)


class LinearDiscriminantAnalysis:
    classes_ = (0, 1, 2)
    n_features_in_ = 4

    def __init__(self, probabilities: list[list[float]] | None = None) -> None:
        self.probabilities = probabilities or [[0.8, 0.1, 0.1]]

    def predict(self, features: object) -> list[int]:
        return [0]

    def predict_proba(self, features: object) -> list[list[float]]:
        return self.probabilities


class StandardScaler:
    def __init__(self, feature_count: int = 4) -> None:
        self.n_features_in_ = feature_count

    def transform(self, features: object) -> object:
        return features


class LabelEncoder:
    def __init__(self, labels: tuple[str, ...]) -> None:
        self.classes_ = labels

    def inverse_transform(self, values: list[int]) -> list[str]:
        return [self.classes_[value] for value in values]


def _bundle_and_manifest(
    model: object | None = None,
    scaler: object | None = None,
    label_encoder: object | None = None,
) -> tuple[bytes, ModelManifest, list[object]]:
    members = {
        "iris_model.pkl": b"model",
        "scaler.pkl": b"scaler",
        "label_encoder.pkl": b"encoder",
        "metadata.pkl": b"metadata",
    }
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    manifest = replace(
        load_manifest(MANIFEST_PATH),
        artifact_checksums={
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in members.items()
        },
    )
    objects = [
        model or LinearDiscriminantAnalysis(),
        scaler or StandardScaler(),
        label_encoder
        or LabelEncoder(("Iris-setosa", "Iris-versicolor", "Iris-virginica")),
    ]
    return stream.getvalue(), manifest, objects


def _load(
    manifest: ModelManifest,
    objects: list[object],
    bundle: bytes,
) -> LoadedIrisModel:
    iterator = iter(objects)
    return load_verified_model(
        bundle, manifest, deserializer=lambda stream: next(iterator)
    )


@pytest.mark.parametrize(
    "manifest_change",
    [
        {"features": ("sepal_width", "sepal_length", "petal_length", "petal_width")},
        {"features": ("sepal_length", "sepal_width", "petal_length")},
        {"classes": ("setosa", "virginica", "versicolor")},
        {"classes": ("setosa", "versicolor", "iris")},
    ],
)
def test_loader_rejects_noncanonical_feature_or_class_schema(manifest_change) -> None:
    bundle, manifest, objects = _bundle_and_manifest()

    with pytest.raises(ArtifactSchemaError, match="schema"):
        _load(replace(manifest, **manifest_change), objects, bundle)


def test_loader_requires_a_four_feature_standard_scaler() -> None:
    bundle, manifest, objects = _bundle_and_manifest(scaler=StandardScaler(3))

    with pytest.raises(ArtifactSchemaError, match="four features"):
        _load(manifest, objects, bundle)


def test_loader_requires_a_four_feature_lda_model() -> None:
    model = LinearDiscriminantAnalysis()
    model.n_features_in_ = 3
    bundle, manifest, objects = _bundle_and_manifest(model=model)

    with pytest.raises(ArtifactSchemaError, match="four features"):
        _load(manifest, objects, bundle)


def test_loader_requires_lda_class_ids_in_label_encoder_order() -> None:
    model = LinearDiscriminantAnalysis()
    model.classes_ = (1, 2, 3)
    bundle, manifest, objects = _bundle_and_manifest(model=model)

    with pytest.raises(ArtifactSchemaError, match="class IDs"):
        _load(manifest, objects, bundle)


def test_loader_requires_source_labels_that_normalize_to_manifest_classes() -> None:
    bundle, manifest, objects = _bundle_and_manifest(
        label_encoder=LabelEncoder(("setosa", "versicolor", "virginica"))
    )

    with pytest.raises(ArtifactSchemaError, match="source labels"):
        _load(manifest, objects, bundle)


def test_loader_requires_predict_and_predict_proba_on_the_lda_model() -> None:
    class MissingProbabilityModel:
        def predict(self, features: object) -> list[int]:
            return [0]

    bundle, manifest, objects = _bundle_and_manifest(model=MissingProbabilityModel())

    with pytest.raises(ArtifactSchemaError, match="predict_proba"):
        _load(manifest, objects, bundle)


@pytest.mark.parametrize("probabilities", [[[1.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]])
def test_prediction_rejects_probability_output_that_is_not_three_classes(
    probabilities: list[list[float]],
) -> None:
    model = LinearDiscriminantAnalysis(probabilities=probabilities)
    bundle, manifest, objects = _bundle_and_manifest(model=model)
    loaded = _load(manifest, objects, bundle)

    with pytest.raises(ArtifactSchemaError, match="three classes"):
        loaded.predict([[5.1, 3.5, 1.4, 0.2]])


def test_prediction_normalizes_source_label_and_returns_three_probabilities() -> None:
    bundle, manifest, objects = _bundle_and_manifest()
    loaded = _load(manifest, objects, bundle)

    prediction = loaded.predict([[5.1, 3.5, 1.4, 0.2]])

    assert prediction.predicted_class == "setosa"
    assert prediction.confidence == 0.8
    assert prediction.probabilities == {
        "setosa": 0.8,
        "versicolor": 0.1,
        "virginica": 0.1,
    }
