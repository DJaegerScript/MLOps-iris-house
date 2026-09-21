"""Tests for training, registration, and promotion release handoff."""

import json
from pathlib import Path

import pytest

import scripts.promote_house_price_model as promotion_script
from house_pricing_mlops.registry import RegistryError, VersionedModelRegistry
from scripts.promote_house_price_model import register_from_summary
from scripts.train_house_price_model import run_training

FIXTURE_PATH = Path("tests/fixtures/house_prices/train.csv")


def _promote_release(*args: object, **kwargs: object) -> object:
    assert hasattr(promotion_script, "promote_release"), (
        "promotion release helper is missing"
    )
    return promotion_script.promote_release(*args, **kwargs)


def _train(tmp_path: Path) -> tuple[dict[str, object], Path]:
    output_dir = tmp_path / "output"
    summary = run_training(
        FIXTURE_PATH,
        dataset_version="fixture-v1",
        output_dir=output_dir,
        tracking_uri=(tmp_path / "mlruns").as_uri(),
        git_commit_sha="abc123",
        random_seed=42,
        dataset_s3_bucket="private-mlops-bucket",
        dataset_s3_key="datasets/house-prices/v1/train.csv",
        dataset_s3_version_id="dataset-version-1",
        model_s3_bucket="private-mlops-bucket",
        model_s3_key="models/house-price/v1/model.tar.gz",
    )
    return summary, output_dir / "training_summary.json"


def test_training_writes_pre_upload_release_draft_without_model_version_id(
    tmp_path: Path,
) -> None:
    summary, summary_path = _train(tmp_path)

    draft_path = summary_path.parent / "release_manifest.json"
    draft = json.loads(draft_path.read_text(encoding="utf-8"))

    assert draft["bundle_sha256"] == summary["bundle_sha256"]
    assert draft["dataset_s3_version_id"] == "dataset-version-1"
    assert draft["model_s3_bucket"] == "private-mlops-bucket"
    assert draft["model_s3_key"] == "models/house-price/v1/model.tar.gz"
    assert draft["model_s3_version_id"] is None
    assert "model.joblib" not in draft
    assert "SalePrice" not in draft


def test_registration_adds_uploaded_model_version_and_creates_candidate(
    tmp_path: Path,
) -> None:
    _, summary_path = _train(tmp_path)
    registry = VersionedModelRegistry(tmp_path / "registry.json")

    candidate = register_from_summary(
        registry,
        summary_path,
        s3_version_id="model-version-1",
    )

    assert candidate["status"] == "candidate"
    assert candidate["model_s3_version_id"] == "model-version-1"
    manifest = registry.read_release_manifest(str(candidate["registry_version"]))
    assert manifest.model_s3_version_id == "model-version-1"
    assert manifest.status == "candidate"


def test_promotion_requires_approval_and_records_it_in_manifest(
    tmp_path: Path,
) -> None:
    _, summary_path = _train(tmp_path)
    registry = VersionedModelRegistry(tmp_path / "registry.json")
    candidate = register_from_summary(
        registry, summary_path, s3_version_id="model-version-1"
    )
    registry_version = str(candidate["registry_version"])

    with pytest.raises(RegistryError, match="approved_by"):
        _promote_release(
            registry, registry_version, approved_by="", reason="reviewed"
        )

    _promote_release(
        registry,
        registry_version,
        approved_by="operator@example.com",
        reason="reviewed metrics",
    )
    promoted = registry.read_release_manifest(registry_version)

    assert promoted.status == "champion"
    assert promoted.approved_by == "operator@example.com"
    assert promoted.approval_reason == "reviewed metrics"
