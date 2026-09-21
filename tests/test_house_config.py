"""Tests for optional House Pricing runtime configuration."""

from dataclasses import fields

import pytest

from house_pricing_mlops.config import HouseSettings, load_house_config

REQUIRED = {
    "HOUSE_MODEL_S3_BUCKET": "private-bucket",
    "HOUSE_MODEL_S3_KEY": "models/house-price/v1/model.tar.gz",
    "HOUSE_MODEL_S3_VERSION_ID": "version-123",
    "HOUSE_MODEL_VERSION": "v1",
    "HOUSE_DATASET_VERSION": "v1",
    "HOUSE_ENVIRONMENT": "test",
    "GIT_COMMIT_SHA": "abc123",
    "DOCKER_IMAGE_VERSION": "image-abc123",
}


def test_house_config_reads_exact_versioned_coordinates() -> None:
    settings = load_house_config(REQUIRED)

    assert settings == HouseSettings(
        model_s3_bucket="private-bucket",
        model_s3_key="models/house-price/v1/model.tar.gz",
        model_s3_version_id="version-123",
        model_version="v1",
        dataset_version="v1",
        environment="test",
        git_commit_sha="abc123",
        docker_image_version="image-abc123",
    )


@pytest.mark.parametrize("missing", sorted(REQUIRED))
def test_house_config_rejects_missing_values(missing: str) -> None:
    environment = dict(REQUIRED)
    environment.pop(missing)

    with pytest.raises(ValueError, match=missing):
        load_house_config(environment)


def test_house_config_has_no_secret_fields() -> None:
    settings = load_house_config(REQUIRED)

    assert [field.name for field in fields(settings)] == [
        "model_s3_bucket",
        "model_s3_key",
        "model_s3_version_id",
        "model_version",
        "dataset_version",
        "environment",
        "git_commit_sha",
        "docker_image_version",
        "max_upload_bytes",
        "max_batch_rows",
    ]
    assert "private" not in repr(settings).lower() or "private-bucket" in repr(settings)
