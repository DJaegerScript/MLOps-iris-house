"""Checksum-verified, in-memory loading of the pinned Iris model bundle."""

from __future__ import annotations

import hashlib
import hmac
import io
import math
import tarfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Protocol

import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder, StandardScaler

from iris_mlops.manifest import ModelManifest, is_trusted_v1_manifest

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
        columns = _probability_columns(probabilities, len(self.classes))

        probability_values: list[float] = []
        for value in columns:
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
            if not 0 <= numeric_value <= 1:
                raise ArtifactSchemaError(
                    "predict_proba probabilities must be within [0, 1]"
                )
            probability_values.append(numeric_value)

        if not math.isclose(
            math.fsum(probability_values), 1.0, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ArtifactSchemaError(
                "predict_proba probabilities must sum approximately to 1"
            )

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
    actual_bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
    if not hmac.compare_digest(actual_bundle_sha256, manifest.bundle_sha256):
        raise ArtifactIntegrityError("bundle checksum mismatch")

    members = _read_verified_members(bundle_bytes, manifest.artifact_checksums)
    load = joblib.load if deserializer is None else deserializer

    try:
        model = load(io.BytesIO(members["iris_model.pkl"]))
        scaler = load(io.BytesIO(members["scaler.pkl"]))
        label_encoder = load(io.BytesIO(members["label_encoder.pkl"]))
    except Exception as error:
        raise ArtifactLoadError("could not deserialize model artifacts") from error
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
    if not is_trusted_v1_manifest(manifest):
        raise ArtifactIntegrityError("manifest does not match trusted v1 manifest")


def _read_verified_members(
    bundle_bytes: bytes,
    checksums: Mapping[str, str],
) -> dict[str, bytes]:
    try:
        archive = tarfile.open(fileobj=io.BytesIO(bundle_bytes), mode="r:gz")
    except Exception as error:
        raise ArtifactLoadError("could not open model bundle archive") from error

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
    except Exception as error:
        raise ArtifactLoadError("could not read model bundle archive") from error

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
    if not isinstance(model, LinearDiscriminantAnalysis):
        raise ArtifactSchemaError(
            "iris_model.pkl must contain a LinearDiscriminantAnalysis"
        )
    if not callable(getattr(model, "predict", None)):
        raise ArtifactSchemaError("LDA model must support predict")
    if not callable(getattr(model, "predict_proba", None)):
        raise ArtifactSchemaError("LDA model must support predict_proba")
    if getattr(model, "n_features_in_", None) != 4:
        raise ArtifactSchemaError("LDA must have four features")
    if not isinstance(scaler, StandardScaler):
        raise ArtifactSchemaError("scaler.pkl must contain a StandardScaler")
    if getattr(scaler, "n_features_in_", None) != 4:
        raise ArtifactSchemaError("StandardScaler must have four features")
    if not callable(getattr(scaler, "transform", None)):
        raise ArtifactSchemaError("StandardScaler must support transform")
    if not isinstance(encoder, LabelEncoder):
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


def _probability_columns(values: object, expected_columns: int) -> list[object]:
    shape_message = (
        "predict_proba must have exact rank-2 shape "
        f"(1, {expected_columns}) with three classes"
    )
    if isinstance(values, (str, bytes, Mapping)):
        raise ArtifactSchemaError(shape_message)
    try:
        rows = list(values)  # type: ignore[arg-type]
    except (AttributeError, TypeError, ValueError) as error:
        raise ArtifactSchemaError(shape_message) from error
    if len(rows) != 1:
        raise ArtifactSchemaError(shape_message)

    row = rows[0]
    if isinstance(row, (str, bytes, Mapping)):
        raise ArtifactSchemaError(shape_message)
    try:
        columns = list(row)  # type: ignore[arg-type]
    except (AttributeError, TypeError, ValueError) as error:
        raise ArtifactSchemaError(shape_message) from error
    if len(columns) != expected_columns:
        raise ArtifactSchemaError(shape_message)
    return columns


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
