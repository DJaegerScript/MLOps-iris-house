"""Strict validation for the metadata-only Iris model manifest."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path

SOURCE = "https://huggingface.co/rajuamburu/iris-classifier"
SOURCE_REVISION = "44b3ab8b94defd12e964f6327fa56c1efd7c4df6"
EXPECTED_FEATURES = (
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
)
EXPECTED_CLASSES = ("setosa", "versicolor", "virginica")
EXPECTED_CLASS_LABEL_NORMALIZATION = {
    "source_labels": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
    "manifest_labels": ["setosa", "versicolor", "virginica"],
    "rule": "Removed the leading 'Iris-' prefix from source labels.",
}
EXPECTED_ARTIFACT_CHECKSUMS = {
    "iris_model.pkl": (
        "7774a99e5a30e7890fa1d75e5e2a44d14718f16b591334fde385d15ad03aa0a7"
    ),
    "scaler.pkl": "10440ff9b1a4b0789ba496b6daffdb803e1bc68b1066a3dd82fcf2a45824daad",
    "label_encoder.pkl": (
        "59c14f9f5ffcd26188226746d0602c8552e0452045e6981138a8e946839ae45c"
    ),
    "metadata.pkl": "7557328ea1eaa3c5e5a6bfd509e538a48f7f9e4a8905a0130f2ba6aa677ad05b",
}
EXPECTED_TRAINING_METRICS = {
    "algorithm": "LDA",
    "probabilities_supported": True,
    "cross_validation": {"folds": 10, "accuracy": 0.975},
    "test_accuracy": 1.0,
    "sample_count": 150,
    "class_count": 3,
    "samples_per_class": 50,
    "provenance": {
        "document": "README.md",
        "source": SOURCE,
        "source_revision": SOURCE_REVISION,
    },
}

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
    "training_metrics",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Validated metadata needed to identify and verify the model bundle."""

    model_name: str
    model_version: str
    source: str
    source_revision: str
    features: tuple[str, ...]
    classes: tuple[str, ...]
    class_label_normalization: dict[str, object]
    framework: str
    framework_version: str
    python_version: str
    artifact_checksums: dict[str, str]
    training_metrics: dict[str, object]


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


def _validate_label_normalization(value: object) -> dict[str, object]:
    field = "class_label_normalization"
    normalization = _require_mapping(value, field)
    _require_exact_keys(
        normalization, ("source_labels", "manifest_labels", "rule"), field
    )
    _require_exact_string_list(
        normalization["source_labels"],
        f"{field}.source_labels",
        ("Iris-setosa", "Iris-versicolor", "Iris-virginica"),
    )
    _require_exact_string_list(
        normalization["manifest_labels"], f"{field}.manifest_labels", EXPECTED_CLASSES
    )
    _require_string(
        normalization["rule"],
        f"{field}.rule",
        EXPECTED_CLASS_LABEL_NORMALIZATION["rule"],
    )
    return deepcopy(dict(normalization))


def _validate_artifact_checksums(value: object) -> dict[str, str]:
    field = "artifact_checksums"
    checksums = _require_mapping(value, field)
    _require_exact_keys(checksums, tuple(EXPECTED_ARTIFACT_CHECKSUMS), field)
    for filename, expected_checksum in EXPECTED_ARTIFACT_CHECKSUMS.items():
        checksum = checksums[filename]
        if not isinstance(checksum, str) or not _SHA256_PATTERN.fullmatch(checksum):
            raise ValueError(f"{field}: {filename} is not a valid sha-256 checksum")
        if checksum != expected_checksum:
            raise ValueError(
                f"{field}: {filename} does not match the verified sha-256 checksum"
            )
    return {filename: checksums[filename] for filename in EXPECTED_ARTIFACT_CHECKSUMS}


def _validate_training_metrics(value: object) -> dict[str, object]:
    field = "training_metrics"
    metrics = _require_mapping(value, field)
    if dict(metrics) != EXPECTED_TRAINING_METRICS:
        raise ValueError(f"{field}: must contain source-reported metrics only")
    return deepcopy(dict(metrics))


def validate_manifest(payload: Mapping[str, object]) -> ModelManifest:
    """Validate a manifest payload and return its typed metadata representation."""

    manifest = _require_mapping(payload, "manifest")
    _require_exact_keys(manifest, _TOP_LEVEL_FIELDS, "manifest")

    return ModelManifest(
        model_name=_require_string(
            manifest["model_name"], "model_name", "iris-classifier"
        ),
        model_version=_require_string(manifest["model_version"], "model_version", "v1"),
        source=_require_string(manifest["source"], "source", SOURCE),
        source_revision=_require_string(
            manifest["source_revision"], "source_revision", SOURCE_REVISION
        ),
        features=_require_exact_string_list(
            manifest["features"], "features", EXPECTED_FEATURES
        ),
        classes=_require_exact_string_list(
            manifest["classes"], "classes", EXPECTED_CLASSES
        ),
        class_label_normalization=_validate_label_normalization(
            manifest["class_label_normalization"]
        ),
        framework=_require_string(manifest["framework"], "framework", "scikit-learn"),
        framework_version=_require_string(
            manifest["framework_version"], "framework_version", "1.8.0"
        ),
        python_version=_require_string(
            manifest["python_version"], "python_version", "not reported by source"
        ),
        artifact_checksums=_validate_artifact_checksums(manifest["artifact_checksums"]),
        training_metrics=_validate_training_metrics(manifest["training_metrics"]),
    )


def load_manifest(path: str | Path) -> ModelManifest:
    """Load and validate a JSON manifest from disk."""

    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, JSONDecodeError) as error:
        raise ValueError(f"manifest: unable to read {manifest_path}") from error
    return validate_manifest(payload)


__all__ = ["ModelManifest", "load_manifest", "validate_manifest"]
