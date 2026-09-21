"""Checksum-verified, in-memory loading of the pinned Iris model bundle."""

from __future__ import annotations

import hashlib
import hmac
import io
import math
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Protocol

import joblib

from iris_mlops.manifest import V1_EXPECTATIONS, ModelManifest

ARTIFACT_NAMES = (
    "iris_model.pkl",
    "scaler.pkl",
    "label_encoder.pkl",
    "metadata.pkl",
)
SOURCE_LABELS = (
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica",
)
NORMALIZED_LABELS = ("setosa", "versicolor", "virginica")


class ArtifactLoadError(ValueError):
    """Base error for malformed or unusable model bundles."""


class ArtifactIntegrityError(ArtifactLoadError):
    """The bundle structure or member checksum is not trusted."""


class ArtifactSchemaError(ArtifactLoadError):
    """A deserialized artifact does not match the Iris v1 schema."""


class ArtifactStore(Protocol):
    """Minimal storage boundary needed by the pure model loader."""

    def get_versioned_object(self, bucket: str, key: str, version_id: str) -> bytes:
        """Return one immutable object version."""


@dataclass(frozen=True, slots=True)
class IrisPrediction:
    """Prediction result with normalized class labels and probabilities."""

    predicted_class: str
    confidence: float
    probabilities: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class LoadedIrisModel:
    """The three inference artifacts allowed by the Iris v1 contract."""

    model: Any
    scaler: Any
    label_encoder: Any
    metadata: object | None = None
    classes: tuple[str, ...] = NORMALIZED_LABELS

    def predict(self, features: object) -> IrisPrediction:
        """Transform one feature row and return a validated Iris prediction."""

        transformed = self.scaler.transform(features)
        predictions = self.model.predict(transformed)
        predicted_value = _first_value(predictions, "predict")

        if isinstance(predicted_value, str):
            source_label = predicted_value
        else:
            try:
                source_label = _first_value(
                    self.label_encoder.inverse_transform([predicted_value]),
                    "label encoder inverse_transform",
                )
            except (AttributeError, IndexError, TypeError, ValueError) as error:
                raise ArtifactSchemaError(
                    "model prediction cannot be decoded by the label encoder"
                ) from error
        predicted_class = _normalize_label(source_label)

        probabilities = self.model.predict_proba(transformed)
        rows = _rows(probabilities)
        if len(rows) != 1 or len(rows[0]) != len(self.classes):
            raise ArtifactSchemaError(
                "predict_proba must return one row with three classes"
            )

        probability_values: list[float] = []
        for value in rows[0]:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as error:
                raise ArtifactSchemaError(
                    "predict_proba must return numeric probabilities"
                ) from error
            if not math.isfinite(numeric_value):
                raise ArtifactSchemaError(
                    "predict_proba must return finite probabilities"
                )
            probability_values.append(numeric_value)

        probability_map = dict(zip(self.classes, probability_values, strict=True))
        return IrisPrediction(
            predicted_class=predicted_class,
            confidence=max(probability_values),
            probabilities=probability_map,
        )


def load_verified_model(
    bundle_bytes: bytes,
    manifest: ModelManifest,
    *,
    deserializer: Callable[[BinaryIO], object] | None = None,
) -> LoadedIrisModel:
    """Verify and load an Iris v1 tar.gz bundle without writing artifact files."""

    _validate_manifest_for_loading(manifest)
    if not isinstance(bundle_bytes, bytes):
        raise ArtifactLoadError("model bundle must be provided as bytes")

    members = _read_verified_members(bundle_bytes, manifest.artifact_checksums)
    load = joblib.load if deserializer is None else deserializer

    model = load(io.BytesIO(members["iris_model.pkl"]))
    scaler = load(io.BytesIO(members["scaler.pkl"]))
    label_encoder = load(io.BytesIO(members["label_encoder.pkl"]))
    _validate_deserialized_artifacts(model, scaler, label_encoder)

    return LoadedIrisModel(
        model=model,
        scaler=scaler,
        label_encoder=label_encoder,
        classes=tuple(manifest.classes),
    )


def load_model_from_s3(
    store: ArtifactStore,
    bucket: str,
    key: str,
    version_id: str,
    manifest: ModelManifest,
    *,
    deserializer: Callable[[BinaryIO], object] | None = None,
) -> LoadedIrisModel:
    """Fetch one explicit S3 object version and pass it to the pure loader."""

    bundle_bytes = store.get_versioned_object(bucket, key, version_id)
    return load_verified_model(
        bundle_bytes,
        manifest,
        deserializer=deserializer,
    )


def _validate_manifest_for_loading(manifest: object) -> None:
    if not isinstance(manifest, ModelManifest):
        raise ArtifactLoadError("a validated v1 manifest is required")

    if tuple(manifest.features) != V1_EXPECTATIONS.features:
        raise ArtifactSchemaError("manifest feature schema is not the Iris v1 schema")
    if tuple(manifest.classes) != NORMALIZED_LABELS:
        raise ArtifactSchemaError("manifest class schema is not the Iris v1 schema")

    normalization = manifest.class_label_normalization
    if (
        tuple(normalization.get("source_labels", ())) != SOURCE_LABELS
        or tuple(normalization.get("manifest_labels", ())) != NORMALIZED_LABELS
    ):
        raise ArtifactSchemaError("manifest source labels do not normalize correctly")

    checksums = manifest.artifact_checksums
    if set(checksums) != set(ARTIFACT_NAMES):
        raise ArtifactIntegrityError("manifest checksums must cover the four artifacts")
    for name in ARTIFACT_NAMES:
        checksum = checksums[name]
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ArtifactIntegrityError(f"manifest checksum for {name} is invalid")
        try:
            int(checksum, 16)
        except ValueError as error:
            raise ArtifactIntegrityError(
                f"manifest checksum for {name} is invalid"
            ) from error


def _read_verified_members(
    bundle_bytes: bytes,
    checksums: Mapping[str, str],
) -> dict[str, bytes]:
    try:
        archive = tarfile.open(fileobj=io.BytesIO(bundle_bytes), mode="r:gz")
    except (tarfile.TarError, OSError) as error:
        raise ArtifactLoadError(
            "model bundle is not a valid gzip tar archive"
        ) from error

    found: dict[str, bytes] = {}
    try:
        with archive:
            for member in archive.getmembers():
                _validate_tar_member(member)
                name = member.name
                if name in found:
                    raise ArtifactIntegrityError(f"duplicate artifact member: {name}")
                if name not in ARTIFACT_NAMES:
                    raise ArtifactIntegrityError(f"unexpected artifact member: {name}")

                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ArtifactIntegrityError(
                        f"could not read artifact member: {name}"
                    )
                found[name] = extracted.read()

    except ArtifactLoadError:
        raise
    except (tarfile.TarError, OSError, EOFError) as error:
        raise ArtifactLoadError("could not read the model bundle") from error

    missing = [name for name in ARTIFACT_NAMES if name not in found]
    if missing:
        raise ArtifactIntegrityError(f"missing artifact members: {', '.join(missing)}")

    for name in ARTIFACT_NAMES:
        actual = hashlib.sha256(found[name]).hexdigest()
        if not hmac.compare_digest(actual, checksums[name]):
            raise ArtifactIntegrityError(f"checksum mismatch for {name}")
    return found


def _validate_tar_member(member: tarfile.TarInfo) -> None:
    name = member.name
    if not name or "\x00" in name:
        raise ArtifactIntegrityError("artifact member has an invalid path")
    if member.issym():
        raise ArtifactIntegrityError(f"symlink artifact member is not allowed: {name}")
    if member.islnk():
        raise ArtifactIntegrityError(f"hardlink artifact member is not allowed: {name}")
    if not member.isfile():
        raise ArtifactIntegrityError(f"unexpected non-file artifact member: {name}")

    posix_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if (
        name.startswith(("/", "\\"))
        or posix_path.is_absolute()
        or windows_path.is_absolute()
    ):
        raise ArtifactIntegrityError(f"absolute path in artifact member: {name}")
    if "\\" in name or ".." in posix_path.parts or ".." in windows_path.parts:
        raise ArtifactIntegrityError(f"path traversal in artifact member: {name}")


def _validate_deserialized_artifacts(
    model: object, scaler: object, encoder: object
) -> None:
    if not callable(getattr(model, "predict", None)):
        raise ArtifactSchemaError("LDA model must support predict")
    if not callable(getattr(model, "predict_proba", None)):
        raise ArtifactSchemaError("LDA model must support predict_proba")
    if type(model).__name__ != "LinearDiscriminantAnalysis":
        raise ArtifactSchemaError(
            "iris_model.pkl must contain a LinearDiscriminantAnalysis"
        )
    if getattr(model, "n_features_in_", None) != 4:
        raise ArtifactSchemaError("LDA must have four features")
    if type(scaler).__name__ != "StandardScaler":
        raise ArtifactSchemaError("scaler.pkl must contain a StandardScaler")
    if getattr(scaler, "n_features_in_", None) != 4:
        raise ArtifactSchemaError("StandardScaler must have four features")
    if not callable(getattr(scaler, "transform", None)):
        raise ArtifactSchemaError("StandardScaler must support transform")
    if type(encoder).__name__ != "LabelEncoder":
        raise ArtifactSchemaError("label_encoder.pkl must contain a LabelEncoder")
    if tuple(getattr(encoder, "classes_", ())) != SOURCE_LABELS:
        raise ArtifactSchemaError(
            "LabelEncoder source labels do not normalize correctly"
        )
    if not callable(getattr(encoder, "inverse_transform", None)):
        raise ArtifactSchemaError("LabelEncoder must support inverse_transform")

    model_classes = getattr(model, "classes_", None)
    try:
        model_class_ids = tuple(model_classes)
    except TypeError as error:
        raise ArtifactSchemaError("LDA class IDs must be (0, 1, 2)") from error
    if model_class_ids != (0, 1, 2):
        raise ArtifactSchemaError("LDA class IDs must be (0, 1, 2)")


def _normalize_label(label: object) -> str:
    if label in SOURCE_LABELS:
        return NORMALIZED_LABELS[SOURCE_LABELS.index(label)]
    if label in NORMALIZED_LABELS:
        return str(label)
    raise ArtifactSchemaError("model returned an unknown Iris source label")


def _first_value(values: object, field: str) -> object:
    if isinstance(values, (str, bytes)):
        return values
    try:
        return values[0]  # type: ignore[index]
    except (IndexError, KeyError, TypeError) as error:
        raise ArtifactSchemaError(f"{field} returned no values") from error


def _rows(values: object) -> list[Sequence[object]]:
    try:
        return list(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ArtifactSchemaError("predict_proba must return rows") from error


__all__ = [
    "ARTIFACT_NAMES",
    "ArtifactIntegrityError",
    "ArtifactLoadError",
    "ArtifactSchemaError",
    "ArtifactStore",
    "IrisPrediction",
    "LoadedIrisModel",
    "load_model_from_s3",
    "load_verified_model",
]
