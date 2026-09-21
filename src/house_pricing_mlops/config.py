"""Optional environment configuration for the House Pricing product."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HouseSettings:
    """Non-secret coordinates and limits required by House serving."""

    model_s3_bucket: str
    model_s3_key: str
    model_s3_version_id: str
    model_version: str
    dataset_version: str
    environment: str
    git_commit_sha: str
    docker_image_version: str
    max_upload_bytes: int = 5 * 1024 * 1024
    max_batch_rows: int = 10_000


_REQUIRED_FIELDS = (
    ("HOUSE_MODEL_S3_BUCKET", "model_s3_bucket"),
    ("HOUSE_MODEL_S3_KEY", "model_s3_key"),
    ("HOUSE_MODEL_S3_VERSION_ID", "model_s3_version_id"),
    ("HOUSE_MODEL_VERSION", "model_version"),
    ("HOUSE_DATASET_VERSION", "dataset_version"),
    ("HOUSE_ENVIRONMENT", "environment"),
    ("GIT_COMMIT_SHA", "git_commit_sha"),
    ("DOCKER_IMAGE_VERSION", "docker_image_version"),
)


def load_house_config(
    environ: Mapping[str, str] | None = None,
) -> HouseSettings:
    """Load allowlisted House configuration only when House is selected."""

    source = os.environ if environ is None else environ
    values: dict[str, str] = {}
    for variable, field_name in _REQUIRED_FIELDS:
        value = source.get(variable)
        if value is None or not value.strip():
            raise ValueError(f"{variable} is required")
        values[field_name] = value.strip()
    return HouseSettings(**values)


__all__ = ["HouseSettings", "load_house_config"]
