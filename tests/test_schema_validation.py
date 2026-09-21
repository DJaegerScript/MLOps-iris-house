"""Tests for validated Iris artifact schemas and inference contracts."""

from __future__ import annotations

import hashlib
import io
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder, StandardScaler

import iris_mlops.model as model_module
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


class StubLinearDiscriminantAnalysis(LinearDiscriminantAnalysis):
    def __init__(self, probabilities: list[list[float]] | None = None) -> None:
        super().__init__()
        self.classes_ = (0, 1, 2)
        self.n_features_in_ = 4
        self.probabilities = (
            probabilities if probabilities is not None else [[0.8, 0.1, 0.1]]
        )

    def predict(self, features: object) -> list[int]:
        return [0]

    def predict_proba(self, features: object) -> list[list[float]]:
        return self.probabilities


class StubStandardScaler(StandardScaler):
    def __init__(self, feature_count: int = 4) -> None:
        super().__init__()
        self.n_features_in_ = feature_count

    def transform(self, features: object) -> object:
        return features


class StubLabelEncoder(LabelEncoder):
    def __init__(self, labels: tuple[str, ...]) -> None:
        super().__init__()
        self.classes_ = labels

    def inverse_transform(self, values: list[int]) -> list[str]:
        return [self.classes_[value] for value in values]


def _bundle_and_manifest(
    model: object | None = None,
    scaler: object | None = None,
    label_encoder: object | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
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
    bundle = stream.getvalue()
    checksums = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in members.items()
    }
    if monkeypatch is not None:
        trusted = replace(
            model_module.V1_EXPECTATIONS,
            artifact_checksums=checksums,
            bundle_sha256=hashlib.sha256(bundle).hexdigest(),
        )
        monkeypatch.setattr(model_module, "V1_EXPECTATIONS", trusted)
    manifest = replace(
        load_manifest(MANIFEST_PATH),
        artifact_checksums=checksums,
        bundle_sha256=hashlib.sha256(bundle).hexdigest(),
    )
    objects = [
        model or StubLinearDiscriminantAnalysis(),
        scaler or StubStandardScaler(),
        label_encoder
        or StubLabelEncoder(
            ("Iris-setosa", "Iris-versicolor", "Iris-virginica")
        ),
    ]
    return bundle, manifest, objects


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
def test_loader_rejects_noncanonical_feature_or_class_schema(
    manifest_change, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest, objects = _bundle_and_manifest(monkeypatch=monkeypatch)

    with pytest.raises(ArtifactSchemaError, match="schema"):
        _load(replace(manifest, **manifest_change), objects, bundle)


def test_loader_requires_a_four_feature_standard_scaler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, manifest, objects = _bundle_and_manifest(
        scaler=StubStandardScaler(3), monkeypatch=monkeypatch
    )

    with pytest.raises(ArtifactSchemaError, match="four features"):
        _load(manifest, objects, bundle)


def test_loader_requires_a_four_feature_lda_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = StubLinearDiscriminantAnalysis()
    model.n_features_in_ = 3
    bundle, manifest, objects = _bundle_and_manifest(
        model=model, monkeypatch=monkeypatch
    )

    with pytest.raises(ArtifactSchemaError, match="four features"):
        _load(manifest, objects, bundle)


def test_loader_requires_lda_class_ids_in_label_encoder_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = StubLinearDiscriminantAnalysis()
    model.classes_ = (1, 2, 3)
    bundle, manifest, objects = _bundle_and_manifest(
        model=model, monkeypatch=monkeypatch
    )

    with pytest.raises(ArtifactSchemaError, match="class IDs"):
        _load(manifest, objects, bundle)


def test_loader_requires_source_labels_that_normalize_to_manifest_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, manifest, objects = _bundle_and_manifest(
        label_encoder=StubLabelEncoder(("setosa", "versicolor", "virginica")),
        monkeypatch=monkeypatch,
    )

    with pytest.raises(ArtifactSchemaError, match="source labels"):
        _load(manifest, objects, bundle)


def test_loader_requires_predict_and_predict_proba_on_the_lda_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = StubLinearDiscriminantAnalysis()
    model.predict_proba = None
    bundle, manifest, objects = _bundle_and_manifest(
        model=model, monkeypatch=monkeypatch
    )

    with pytest.raises(ArtifactSchemaError, match="predict_proba"):
        _load(manifest, objects, bundle)


@pytest.mark.parametrize(
    "probabilities",
    [
        [[1.0, 0.0]],
        [[1.0, 0.0, 0.0, 0.0]],
        [1.0],
        [[[1.0, 0.0, 0.0]]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    ],
)
def test_prediction_rejects_probability_output_that_is_not_three_classes(
    probabilities: list[list[float]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = StubLinearDiscriminantAnalysis(probabilities=probabilities)
    bundle, manifest, objects = _bundle_and_manifest(
        model=model, monkeypatch=monkeypatch
    )
    loaded = _load(manifest, objects, bundle)

    with pytest.raises(ArtifactSchemaError, match="three classes"):
        loaded.predict([[5.1, 3.5, 1.4, 0.2]])


def test_prediction_normalizes_source_label_and_returns_three_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, manifest, objects = _bundle_and_manifest(monkeypatch=monkeypatch)
    loaded = _load(manifest, objects, bundle)

    prediction = loaded.predict([[5.1, 3.5, 1.4, 0.2]])

    assert prediction.predicted_class == "setosa"
    assert prediction.confidence == 0.8
    assert prediction.probabilities == {
        "setosa": 0.8,
        "versicolor": 0.1,
        "virginica": 0.1,
    }
