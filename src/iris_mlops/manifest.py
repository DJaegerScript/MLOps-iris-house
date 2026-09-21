"""Strict validation for the metadata-only iris-classifier v1 manifest."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from types import MappingProxyType

_TOP_LEVEL_FIELDS = (
    "model_name",
    "model_version",
    "source",
    "source_revision",
    "features",
    "classes",
    "class_label_normalization",
    "framework",
    "framework_version",
    "python_version",
    "artifact_checksums",
    "bundle_sha256",
    "training_metrics",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ManifestLoadError(ValueError):
    """Base error for failures while reading a manifest file."""


class ManifestNotFoundError(ManifestLoadError):
    """The requested manifest file does not exist."""


class ManifestEncodingError(ManifestLoadError):
    """The manifest file is not valid UTF-8."""


class ManifestJSONError(ManifestLoadError):
    """The manifest file is not valid JSON."""


class ManifestReadError(ManifestLoadError):
    """The manifest file could not be read for another I/O reason."""


@dataclass(frozen=True, slots=True)
class V1ManifestExpectations:
    """Trusted, version-specific expectations for the iris-classifier v1 contract."""

    validator_name: str
    model_name: str
    model_version: str
    source: str
    source_revision: str
    features: tuple[str, ...]
    classes: tuple[str, ...]
    class_label_normalization: Mapping[str, object]
    framework: str
    framework_version: str
    python_version: str
    artifact_checksums: Mapping[str, str]
    bundle_sha256: str
    training_metrics: Mapping[str, object]


V1_EXPECTATIONS = V1ManifestExpectations(
    validator_name="iris-classifier-v1",
    model_name="iris-classifier",
    model_version="v1",
    source="https://huggingface.co/rajuamburu/iris-classifier",
    source_revision="44b3ab8b94defd12e964f6327fa56c1efd7c4df6",
    features=("sepal_length", "sepal_width", "petal_length", "petal_width"),
    classes=("setosa", "versicolor", "virginica"),
    class_label_normalization=MappingProxyType(
        {
            "source_labels": (
                "Iris-setosa",
                "Iris-versicolor",
                "Iris-virginica",
            ),
            "manifest_labels": ("setosa", "versicolor", "virginica"),
            "rule": "Removed the leading 'Iris-' prefix from source labels.",
        }
    ),
    framework="scikit-learn",
    framework_version="1.8.0",
    python_version="not reported by source",
    artifact_checksums=MappingProxyType(
        {
            "iris_model.pkl": (
                "7774a99e5a30e7890fa1d75e5e2a44d14718f16b591334fde385d15ad03aa0a7"
            ),
            "scaler.pkl": (
                "10440ff9b1a4b0789ba496b6daffdb803e1bc68b1066a3dd82fcf2a45824daad"
            ),
            "label_encoder.pkl": (
                "59c14f9f5ffcd26188226746d0602c8552e0452045e6981138a8e946839ae45c"
            ),
            "metadata.pkl": (
                "7557328ea1eaa3c5e5a6bfd509e538a48f7f9e4a8905a0130f2ba6aa677ad05b"
            ),
        }
    ),
    bundle_sha256=(
        "a0da1d85be416055f09551cd3ef1070e21503a24f675c2dfd7077b27262a35dd"
    ),
    training_metrics=MappingProxyType(
        {
            "algorithm": "LDA",
            "probabilities_supported": True,
            "cross_validation": MappingProxyType(
                {"folds": 10, "accuracy": 0.975}
            ),
            "test_accuracy": 1.0,
            "sample_count": 150,
            "class_count": 3,
            "samples_per_class": 50,
            "provenance": MappingProxyType(
                {
                    "document": "README.md",
                    "source": "https://huggingface.co/rajuamburu/iris-classifier",
                    "source_revision": (
                        "44b3ab8b94defd12e964f6327fa56c1efd7c4df6"
                    ),
                }
            ),
        }
    ),
)
V1_VALIDATOR_NAME = V1_EXPECTATIONS.validator_name


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Validated metadata needed to identify and verify the v1 model bundle."""

    model_name: str
    model_version: str
    source: str
    source_revision: str
    features: tuple[str, ...]
    classes: tuple[str, ...]
    class_label_normalization: Mapping[str, object]
    framework: str
    framework_version: str
    python_version: str
    artifact_checksums: Mapping[str, str]
    bundle_sha256: str
    training_metrics: Mapping[str, object]


def is_trusted_v1_manifest(manifest: ModelManifest) -> bool:
    """Return whether every validated manifest field matches the v1 contract."""

    if not isinstance(manifest, ModelManifest):
        return False
    expectations = V1_EXPECTATIONS
    return (
        manifest.model_name == expectations.model_name
        and manifest.model_version == expectations.model_version
        and manifest.source == expectations.source
        and manifest.source_revision == expectations.source_revision
        and manifest.features == expectations.features
        and manifest.classes == expectations.classes
        and manifest.class_label_normalization
        == expectations.class_label_normalization
        and manifest.framework == expectations.framework
        and manifest.framework_version == expectations.framework_version
        and manifest.python_version == expectations.python_version
        and manifest.artifact_checksums == expectations.artifact_checksums
        and manifest.bundle_sha256 == expectations.bundle_sha256
        and manifest.training_metrics == expectations.training_metrics
    )


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field}: expected an object")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: tuple[str, ...], field: str
) -> None:
    actual = set(value)
    expected_keys = set(expected)
    missing = expected_keys - actual
    unexpected = actual - expected_keys
    if missing:
        missing_field = sorted(missing)[0]
        raise ValueError(f"{field}: missing required field {missing_field}")
    if unexpected:
        unexpected_field = sorted(unexpected)[0]
        raise ValueError(f"{field}: unexpected field {unexpected_field}")


def _require_string(value: object, field: str, expected: str | None = None) -> str:
    if not isinstance(value, str) or (expected is not None and value != expected):
        if expected is None:
            raise ValueError(f"{field}: expected a string")
        raise ValueError(f"{field}: expected {expected!r}")
    return value


def _require_exact_string_list(
    value: object, field: str, expected: tuple[str, ...]
) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field}: expected a list of strings")
    if tuple(value) != expected:
        raise ValueError(f"{field}: does not match the required ordered values")
    return tuple(value)


def _freeze_metadata(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_metadata(nested) for key, nested in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(nested) for nested in value)
    return value


def _strict_json_equal(actual: object, expected: object) -> bool:
    """Compare JSON values without treating bools and numbers as interchangeable."""

    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
            return False
        if set(actual) != set(expected):
            return False
        return all(
            _strict_json_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or not isinstance(
            expected, (list, tuple)
        ):
            return False
        return len(actual) == len(expected) and all(
            _strict_json_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return type(actual) is type(expected) and actual == expected


def _validate_label_normalization(value: object) -> Mapping[str, object]:
    field = "class_label_normalization"
    normalization = _require_mapping(value, field)
    _require_exact_keys(
        normalization, ("source_labels", "manifest_labels", "rule"), field
    )
    _require_exact_string_list(
        normalization["source_labels"],
        f"{field}.source_labels",
        V1_EXPECTATIONS.class_label_normalization["source_labels"],
    )
    _require_exact_string_list(
        normalization["manifest_labels"],
        f"{field}.manifest_labels",
        V1_EXPECTATIONS.classes,
    )
    _require_string(
        normalization["rule"],
        f"{field}.rule",
        V1_EXPECTATIONS.class_label_normalization["rule"],
    )
    return _freeze_metadata(normalization)


def _validate_artifact_checksums(value: object) -> Mapping[str, str]:
    field = "artifact_checksums"
    checksums = _require_mapping(value, field)
    _require_exact_keys(checksums, tuple(V1_EXPECTATIONS.artifact_checksums), field)
    for filename, expected_checksum in V1_EXPECTATIONS.artifact_checksums.items():
        checksum = checksums[filename]
        if not isinstance(checksum, str) or not _SHA256_PATTERN.fullmatch(checksum):
            raise ValueError(f"{field}: {filename} is not a valid sha-256 checksum")
        if checksum != expected_checksum:
            raise ValueError(
                f"{field}: {filename} does not match the verified sha-256 checksum"
            )
    return MappingProxyType(
        {
            filename: checksums[filename]
            for filename in V1_EXPECTATIONS.artifact_checksums
        }
    )


def _validate_bundle_sha256(value: object) -> str:
    field = "bundle_sha256"
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field}: is not a valid sha-256 checksum")
    if value != V1_EXPECTATIONS.bundle_sha256:
        raise ValueError(f"{field}: does not match the verified sha-256 checksum")
    return value


def _validate_training_metrics(value: object) -> Mapping[str, object]:
    field = "training_metrics"
    metrics = _require_mapping(value, field)
    if not _strict_json_equal(metrics, V1_EXPECTATIONS.training_metrics):
        raise ValueError(f"{field}: must contain source-reported metrics only")
    return _freeze_metadata(metrics)


def validate_v1_manifest(payload: Mapping[str, object]) -> ModelManifest:
    """Validate the metadata-only iris-classifier v1 manifest contract."""

    manifest = _require_mapping(payload, "manifest")
    _require_exact_keys(manifest, _TOP_LEVEL_FIELDS, "manifest")
    expectations = V1_EXPECTATIONS

    return ModelManifest(
        model_name=_require_string(
            manifest["model_name"], "model_name", expectations.model_name
        ),
        model_version=_require_string(
            manifest["model_version"], "model_version", expectations.model_version
        ),
        source=_require_string(manifest["source"], "source", expectations.source),
        source_revision=_require_string(
            manifest["source_revision"],
            "source_revision",
            expectations.source_revision,
        ),
        features=_require_exact_string_list(
            manifest["features"], "features", expectations.features
        ),
        classes=_require_exact_string_list(
            manifest["classes"], "classes", expectations.classes
        ),
        class_label_normalization=_validate_label_normalization(
            manifest["class_label_normalization"]
        ),
        framework=_require_string(
            manifest["framework"], "framework", expectations.framework
        ),
        framework_version=_require_string(
            manifest["framework_version"],
            "framework_version",
            expectations.framework_version,
        ),
        python_version=_require_string(
            manifest["python_version"],
            "python_version",
            expectations.python_version,
        ),
        artifact_checksums=_validate_artifact_checksums(manifest["artifact_checksums"]),
        bundle_sha256=_validate_bundle_sha256(manifest["bundle_sha256"]),
        training_metrics=_validate_training_metrics(manifest["training_metrics"]),
    )


def validate_manifest(payload: Mapping[str, object]) -> ModelManifest:
    """Validate using the explicitly supported iris-classifier v1 contract."""

    return validate_v1_manifest(payload)


def load_manifest(path: str | Path) -> ModelManifest:
    """Load and validate the metadata-only iris-classifier v1 manifest."""

    manifest_path = Path(path)
    try:
        manifest_text = manifest_path.read_bytes().decode("utf-8")
    except FileNotFoundError as error:
        raise ManifestNotFoundError(
            f"manifest file not found: {manifest_path}"
        ) from error
    except UnicodeDecodeError as error:
        raise ManifestEncodingError(
            f"manifest contains invalid UTF-8: {manifest_path}"
        ) from error
    except OSError as error:
        raise ManifestReadError(
            f"manifest could not be read: {manifest_path}"
        ) from error

    try:
        payload = json.loads(manifest_text)
    except JSONDecodeError as error:
        raise ManifestJSONError(
            f"manifest contains malformed JSON: {manifest_path}"
        ) from error
    return validate_v1_manifest(payload)


__all__ = [
    "ManifestEncodingError",
    "ManifestJSONError",
    "ManifestLoadError",
    "ManifestNotFoundError",
    "ManifestReadError",
    "ModelManifest",
    "V1_EXPECTATIONS",
    "V1_VALIDATOR_NAME",
    "V1ManifestExpectations",
    "is_trusted_v1_manifest",
    "load_manifest",
    "validate_manifest",
    "validate_v1_manifest",
]
