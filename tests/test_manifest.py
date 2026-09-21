"""Tests for the metadata-only Iris model manifest."""

import importlib
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "artifacts" / "iris-classifier-v1.manifest.json"
MANIFEST_MODULE_PATH = REPOSITORY_ROOT / "src" / "iris_mlops" / "manifest.py"
SOURCE_URL = "https://huggingface.co/rajuamburu/iris-classifier"
SOURCE_REVISION = "44b3ab8b94defd12e964f6327fa56c1efd7c4df6"
EXPECTED_FEATURES = (
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
)
EXPECTED_CLASSES = ("setosa", "versicolor", "virginica")
EXPECTED_LABEL_NORMALIZATION = {
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
EXPECTED_BUNDLE_SHA256 = (
    "a0da1d85be416055f09551cd3ef1070e21503a24f675c2dfd7077b27262a35dd"
)
REQUIRED_FIELDS = (
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


@pytest.fixture
def manifest_api():
    """Load the implementation after the RED phase has established its absence."""

    assert MANIFEST_MODULE_PATH.is_file(), "manifest implementation is missing"
    return importlib.import_module("iris_mlops.manifest")


def _manifest_payload() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text())


def test_manifest_loads_the_complete_metadata_contract() -> None:
    assert MANIFEST_MODULE_PATH.is_file(), "manifest implementation is missing"
    manifest_api = importlib.import_module("iris_mlops.manifest")
    manifest = manifest_api.load_manifest(MANIFEST_PATH)

    assert manifest.model_name == "iris-classifier"
    assert manifest.model_version == "v1"
    assert manifest.source == SOURCE_URL
    assert manifest.source_revision == SOURCE_REVISION
    assert manifest.features == EXPECTED_FEATURES
    assert manifest.classes == EXPECTED_CLASSES
    assert dict(manifest.class_label_normalization) == {
        "source_labels": tuple(EXPECTED_LABEL_NORMALIZATION["source_labels"]),
        "manifest_labels": tuple(EXPECTED_LABEL_NORMALIZATION["manifest_labels"]),
        "rule": EXPECTED_LABEL_NORMALIZATION["rule"],
    }
    assert manifest.framework == "scikit-learn"
    assert manifest.framework_version == "1.8.0"
    assert manifest.python_version == "not reported by source"
    assert manifest.artifact_checksums == EXPECTED_ARTIFACT_CHECKSUMS
    assert manifest.bundle_sha256 == EXPECTED_BUNDLE_SHA256
    assert manifest.training_metrics == {
        "algorithm": "LDA",
        "probabilities_supported": True,
        "cross_validation": {"folds": 10, "accuracy": 0.975},
        "test_accuracy": 1.0,
        "sample_count": 150,
        "class_count": 3,
        "samples_per_class": 50,
        "provenance": {
            "document": "README.md",
            "source": SOURCE_URL,
            "source_revision": SOURCE_REVISION,
        },
    }


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_manifest_requires_every_contract_field(manifest_api, field: str) -> None:
    payload = _manifest_payload()
    payload.pop(field)

    with pytest.raises(ValueError, match=field):
        manifest_api.validate_manifest(payload)


def test_manifest_rejects_unknown_top_level_metadata(manifest_api) -> None:
    payload = _manifest_payload()
    payload["unexpected"] = "not part of the contract"

    with pytest.raises(ValueError, match="unexpected"):
        manifest_api.validate_manifest(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("features", ["sepal_width", "sepal_length", "petal_length", "petal_width"]),
        ("features", [*EXPECTED_FEATURES, "extra_feature"]),
        ("classes", ["setosa", "virginica", "versicolor"]),
        ("classes", ["setosa", "versicolor", "other"]),
    ],
)
def test_manifest_rejects_noncanonical_schema(manifest_api, field, value) -> None:
    payload = _manifest_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        manifest_api.validate_manifest(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", "https://example.com/another-model"),
        ("source_revision", "main"),
        ("model_version", "latest"),
        ("framework", "tensorflow"),
    ],
)
def test_manifest_rejects_untrusted_or_unsupported_identity(
    manifest_api, field, value
) -> None:
    payload = _manifest_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        manifest_api.validate_manifest(payload)


@pytest.mark.parametrize(
    "checksum",
    [
        "",
        "not-a-sha256",
        "0" * 63,
        "g" * 64,
    ],
)
def test_manifest_requires_valid_sha256_artifact_checksums(
    manifest_api, checksum: str
) -> None:
    payload = _manifest_payload()
    payload["artifact_checksums"]["iris_model.pkl"] = checksum

    with pytest.raises(ValueError, match="sha-256"):
        manifest_api.validate_manifest(payload)


@pytest.mark.parametrize("checksum", ["", "not-a-sha256", "0" * 63, "g" * 64])
def test_manifest_requires_valid_sha256_bundle_checksum(
    manifest_api, checksum: str
) -> None:
    payload = _manifest_payload()
    payload["bundle_sha256"] = checksum

    with pytest.raises(ValueError, match="sha-256"):
        manifest_api.validate_manifest(payload)


def test_manifest_requires_exact_verified_artifact_set(manifest_api) -> None:
    payload = _manifest_payload()
    del payload["artifact_checksums"]["metadata.pkl"]

    with pytest.raises(ValueError, match="artifact_checksums"):
        manifest_api.validate_manifest(payload)


def test_training_metrics_are_explicitly_source_reported(manifest_api) -> None:
    payload = _manifest_payload()
    metrics = payload["training_metrics"]

    assert metrics["provenance"] == {
        "document": "README.md",
        "source": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
    }
    assert metrics["algorithm"] == "LDA"
    assert metrics["probabilities_supported"] is True
    assert metrics["cross_validation"] == {"folds": 10, "accuracy": 0.975}
    assert metrics["test_accuracy"] == 1.0
    assert metrics["sample_count"] == 150
    assert metrics["class_count"] == 3
    assert metrics["samples_per_class"] == 50

    metrics["invented_runtime_metric"] = 0.99
    with pytest.raises(ValueError, match="training_metrics"):
        manifest_api.validate_manifest(payload)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("probabilities_supported",), 1),
        (("cross_validation", "folds"), True),
        (("test_accuracy",), 1),
    ],
)
def test_training_metrics_require_exact_json_scalar_types(
    manifest_api, path: tuple[str, ...], replacement: object
) -> None:
    payload = _manifest_payload()
    value = payload["training_metrics"]
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement

    with pytest.raises(ValueError, match="training_metrics"):
        manifest_api.validate_manifest(payload)


def test_v1_validator_is_explicitly_named_and_versioned(manifest_api) -> None:
    assert manifest_api.V1_VALIDATOR_NAME == "iris-classifier-v1"
    manifest = manifest_api.validate_v1_manifest(_manifest_payload())

    assert manifest.model_name == "iris-classifier"
    assert manifest.model_version == "v1"


def test_manifest_documents_source_label_normalization() -> None:
    payload = _manifest_payload()

    assert payload["classes"] == list(EXPECTED_CLASSES)
    assert payload["class_label_normalization"] == EXPECTED_LABEL_NORMALIZATION


def test_load_manifest_distinguishes_missing_file(manifest_api, tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(
        manifest_api.ManifestNotFoundError, match="manifest file not found"
    ):
        manifest_api.load_manifest(missing_path)


def test_load_manifest_distinguishes_malformed_json(
    manifest_api, tmp_path: Path
) -> None:
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(manifest_api.ManifestJSONError, match="malformed JSON"):
        manifest_api.load_manifest(malformed_path)


def test_load_manifest_distinguishes_invalid_utf8(manifest_api, tmp_path: Path) -> None:
    invalid_utf8_path = tmp_path / "invalid-utf8.json"
    invalid_utf8_path.write_bytes(b"\xff\xfe")

    with pytest.raises(manifest_api.ManifestEncodingError, match="invalid UTF-8"):
        manifest_api.load_manifest(invalid_utf8_path)


def test_validated_nested_metadata_is_immutable(manifest_api) -> None:
    manifest = manifest_api.load_manifest(MANIFEST_PATH)

    with pytest.raises(TypeError):
        manifest.artifact_checksums["metadata.pkl"] = "changed"
    with pytest.raises(TypeError):
        manifest.class_label_normalization["rule"] = "changed"
    with pytest.raises(TypeError):
        manifest.class_label_normalization["manifest_labels"][0] = "changed"
    with pytest.raises(TypeError):
        manifest.training_metrics["test_accuracy"] = 0.5
    with pytest.raises(TypeError):
        manifest.training_metrics["cross_validation"]["folds"] = 5


def test_manifest_contains_metadata_only(manifest_api) -> None:
    payload = _manifest_payload()
    serialized = json.dumps(payload)

    assert set(payload) == set(REQUIRED_FIELDS)
    assert "dataset" not in serialized.lower()
    assert "credentials" not in serialized.lower()
    assert "secret" not in serialized.lower()
    assert "model_bytes" not in serialized.lower()
    assert all(len(value) < 1024 for value in serialized.split('"') if value)

    def assert_metadata(value: object) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                assert isinstance(key, str)
                assert not any(
                    forbidden in key.lower()
                    for forbidden in ("credential", "secret", "dataset", "row", "bytes")
                )
                assert_metadata(nested)
        elif isinstance(value, list):
            for nested in value:
                assert_metadata(nested)
        else:
            assert isinstance(value, (bool, float, int, str))

    assert_metadata(payload)
