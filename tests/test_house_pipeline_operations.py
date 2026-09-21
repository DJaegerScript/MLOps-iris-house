"""Tests for safe House pipeline smoke-test and rollback operations."""

import json

import pytest

from house_pricing_mlops.release import HouseReleaseManifest
from scripts.codepipeline.rollback_house_release import validate_rollback_arguments
from scripts.codepipeline.smoke_test_house_release import (
    build_safe_batch_csv,
    build_smoke_summary,
    validate_release_coordinates,
)


def _release() -> HouseReleaseManifest:
    return HouseReleaseManifest.from_dict(
        {
            "model_name": "house-price-model",
            "model_version": "v2",
            "registry_version": "7",
            "dataset_version": "v3",
            "dataset_s3_bucket": "private-bucket",
            "dataset_s3_key": "datasets/house-prices/v3/train.csv",
            "dataset_s3_version_id": "dataset-version-3",
            "model_s3_bucket": "private-bucket",
            "model_s3_key": "models/house-price/v2/model.tar.gz",
            "model_s3_version_id": "model-version-2",
            "bundle_sha256": "a" * 64,
            "mlflow_run_id": "run-7",
            "selected_model": "gradient_boosting",
            "metrics": {"rmse": 0.12},
            "git_commit_sha": "abc123",
            "status": "champion",
            "approved_by": "codepipeline/automatic",
            "approval_reason": "automatic promotion",
        }
    )


def test_smoke_coordinates_and_report_are_safe_and_complete() -> None:
    image = {
        "image_uri": "123.dkr.ecr.example/iris-mlops:abc123",
        "commit_sha": "abc123",
        "digest": "sha256:" + "b" * 64,
    }

    coordinates = validate_release_coordinates(image, _release())
    summary = build_smoke_summary(
        image,
        _release(),
        health_status="ok",
        latency_ms=12.5,
        checks={"iris_sample": "passed", "house_online": "passed"},
    )

    assert coordinates["house_model_s3_version_id"] == "model-version-2"
    assert summary["image_digest"] == image["digest"]
    assert summary["house_model_version"] == "v2"
    assert summary["health_status"] == "ok"
    assert "Address" not in json.dumps(summary)
    assert "SalePrice" not in json.dumps(summary)


def test_safe_batch_fixtures_cover_labeled_and_unlabeled_contracts() -> None:
    labeled = build_safe_batch_csv(labeled=True)
    unlabeled = build_safe_batch_csv(labeled=False)

    assert b"SalePrice" in labeled
    assert b"SalePrice" not in unlabeled
    assert b"Address" not in labeled
    assert b"identifier" not in labeled.lower()


def test_rollback_requires_prior_immutable_image_and_model_release() -> None:
    with pytest.raises(ValueError, match="image"):
        validate_rollback_arguments(
            previous_image_uri="latest",
            previous_image_digest="sha256:" + "b" * 64,
            previous_release=_release(),
            reason="restore known good release",
        )

    rollback = validate_rollback_arguments(
        previous_image_uri="123.dkr.ecr.example/iris-mlops:abc123",
        previous_image_digest="sha256:" + "b" * 64,
        previous_release=_release(),
        reason="restore known good release",
    )

    assert rollback["image_uri"].endswith(":abc123")
    assert rollback["image_digest"].startswith("sha256:")
    assert rollback["house_model_s3_version_id"] == "model-version-2"
    assert rollback["reason"] == "restore known good release"
