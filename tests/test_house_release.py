"""Tests for the immutable House Pricing release-manifest contract."""

import importlib
import json
from pathlib import Path

import pytest

from house_pricing_mlops.registry import VersionedModelRegistry

RELEASE_MODULE_PATH = Path(__file__).parents[1] / "src/house_pricing_mlops/release.py"


def _release_api():
    assert RELEASE_MODULE_PATH.is_file(), "release implementation is missing"
    return importlib.import_module("house_pricing_mlops.release")


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_name": "house-price-model",
        "model_version": "v2",
        "registry_version": "7",
        "dataset_version": "v3",
        "dataset_s3_bucket": "private-mlops-bucket",
        "dataset_s3_key": "datasets/house-prices/v3/train.csv",
        "dataset_s3_version_id": "dataset-object-version-3",
        "model_s3_bucket": "private-mlops-bucket",
        "model_s3_key": "models/house-price/v2/model.tar.gz",
        "model_s3_version_id": "model-object-version-2",
        "bundle_sha256": "a" * 64,
        "mlflow_run_id": "run-123",
        "selected_model": "gradient_boosting",
        "metrics": {"rmse": 0.12, "rmsle": 0.04},
        "git_commit_sha": "abc123",
        "status": "candidate",
    }
    payload.update(overrides)
    return payload


def test_release_manifest_is_json_safe_and_deterministic() -> None:
    HouseReleaseManifest = _release_api().HouseReleaseManifest
    manifest = HouseReleaseManifest.from_dict(_payload())

    first = manifest.to_json()
    second = manifest.to_json()

    assert first == second
    assert json.loads(first) == _payload()
    assert "AWS_ACCESS_KEY_ID" not in first
    assert "train.csv content" not in first


@pytest.mark.parametrize(
    "field",
    [
        "model_name",
        "model_version",
        "registry_version",
        "dataset_version",
        "dataset_s3_bucket",
        "dataset_s3_key",
        "dataset_s3_version_id",
        "model_s3_bucket",
        "model_s3_key",
        "model_s3_version_id",
        "bundle_sha256",
        "mlflow_run_id",
        "selected_model",
        "metrics",
        "git_commit_sha",
        "status",
    ],
)
def test_release_manifest_rejects_missing_required_fields(field: str) -> None:
    release_api = _release_api()
    payload = _payload()
    payload.pop(field)

    with pytest.raises(release_api.ReleaseValidationError, match=field):
        release_api.HouseReleaseManifest.from_dict(payload)


@pytest.mark.parametrize(
    "field",
    [
        "dataset_s3_version_id",
        "model_s3_version_id",
        "dataset_s3_key",
        "model_s3_key",
    ],
)
def test_release_manifest_rejects_missing_or_mutable_s3_coordinates(field: str) -> None:
    release_api = _release_api()
    missing = _payload(**{field: ""})
    with pytest.raises(release_api.ReleaseValidationError, match=field):
        release_api.HouseReleaseManifest.from_dict(missing)

    mutable = _payload(**{field: "latest"})
    with pytest.raises(release_api.ReleaseValidationError, match="latest"):
        release_api.HouseReleaseManifest.from_dict(mutable)


@pytest.mark.parametrize("status", ["pending", "unapproved", "approved", "latest"])
def test_release_manifest_rejects_unapproved_statuses(status: str) -> None:
    release_api = _release_api()
    with pytest.raises(release_api.ReleaseValidationError, match="status"):
        release_api.HouseReleaseManifest.from_dict(_payload(status=status))


def test_promoted_release_requires_and_records_approval_metadata() -> None:
    release_api = _release_api()
    with pytest.raises(release_api.ReleaseValidationError, match="approved_by"):
        release_api.HouseReleaseManifest.from_dict(_payload(status="champion"))

    manifest = release_api.HouseReleaseManifest.from_dict(
        _payload(
            status="champion",
            approved_by="operator@example.com",
            approval_reason="reviewed metrics",
        )
    )

    assert manifest.to_dict()["approved_by"] == "operator@example.com"
    assert manifest.to_dict()["approval_reason"] == "reviewed metrics"


def test_registry_creates_and_promotes_a_release_manifest(tmp_path: Path) -> None:
    registry = VersionedModelRegistry(tmp_path / "registry.json")

    candidate = registry.register_candidate_release(
        model_name="house-price-model",
        model_version="v2",
        run_id="run-123",
        dataset_version="v3",
        dataset_s3_bucket="private-mlops-bucket",
        dataset_s3_key="datasets/house-prices/v3/train.csv",
        dataset_s3_version_id="dataset-object-version-3",
        bundle_sha256="a" * 64,
        model_s3_bucket="private-mlops-bucket",
        model_s3_key="models/house-price/v2/model.tar.gz",
        model_s3_version_id="model-object-version-2",
        selected_model="gradient_boosting",
        metrics={"rmse": 0.12},
        git_commit_sha="abc123",
    )

    assert candidate.status == "candidate"
    assert registry.read_release_manifest(candidate.registry_version) == candidate

    promoted = registry.promote_champion(
        candidate.registry_version,
        approved_by="operator@example.com",
        reason="reviewed metrics",
    )

    assert promoted["status"] == "champion"
    promoted_manifest = registry.read_release_manifest(candidate.registry_version)
    assert promoted_manifest.status == "champion"
    assert promoted_manifest.approved_by == "operator@example.com"
    assert promoted_manifest.approval_reason == "reviewed metrics"
