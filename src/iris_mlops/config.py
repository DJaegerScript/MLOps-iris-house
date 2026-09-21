"""Environment-driven configuration for the Iris application."""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    """Non-secret runtime settings required by the Iris application."""

    model_s3_bucket: str
    model_s3_key: str
    model_s3_version_id: str
    model_version: str
    environment: str
    git_commit_sha: str
    docker_image_version: str
    low_confidence_threshold: float


_ENVIRONMENT_FIELDS = (
    ("IRIS_MODEL_S3_BUCKET", "model_s3_bucket"),
    ("IRIS_MODEL_S3_KEY", "model_s3_key"),
    ("IRIS_MODEL_S3_VERSION_ID", "model_s3_version_id"),
    ("IRIS_MODEL_VERSION", "model_version"),
    ("IRIS_ENVIRONMENT", "environment"),
    ("GIT_COMMIT_SHA", "git_commit_sha"),
    ("DOCKER_IMAGE_VERSION", "docker_image_version"),
)


def _required_value(environment: Mapping[str, str], variable: str) -> str:
    value = environment.get(variable)
    if value is None or not value.strip():
        raise ValueError(f"{variable} is required")
    return value.strip()


def _confidence_threshold(environment: Mapping[str, str]) -> float:
    variable = "LOW_CONFIDENCE_THRESHOLD"
    raw_value = _required_value(environment, variable)

    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(
            f"{variable} must be a finite number between 0 and 1"
        ) from error

    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{variable} must be a finite number between 0 and 1")
    return value


def load_config(environ: Mapping[str, str] | None = None) -> Settings:
    """Load the allowlisted, non-secret application settings from the environment."""

    environment = os.environ if environ is None else environ
    values = {
        field_name: _required_value(environment, variable)
        for variable, field_name in _ENVIRONMENT_FIELDS
    }
    values["low_confidence_threshold"] = _confidence_threshold(environment)
    return Settings(**values)
