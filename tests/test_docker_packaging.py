"""Static and unit checks for the application-only Docker packaging."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_is_pinned_non_root_and_health_checked() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "FROM python:3.12-slim-bookworm" in dockerfile
    assert "COPY requirements.txt" in dockerfile
    assert "pip install --no-cache-dir" in dockerfile
    assert "EXPOSE 8501" in dockerfile
    assert "USER iris" in dockerfile
    assert "--server.address=0.0.0.0" in dockerfile
    assert "--server.port=8501" in dockerfile
    assert "/_stcore/health" in dockerfile
    assert re.search(r"HEALTHCHECK .*CMD", dockerfile, re.DOTALL)
    assert "AWS_ACCESS_KEY_ID" not in dockerfile
    assert "AWS_SECRET_ACCESS_KEY" not in dockerfile
    assert "aws s3" not in dockerfile.lower()


@pytest.mark.parametrize(
    "entry",
    [
        ".env",
        ".env.*",
        "*.pkl",
        "*.joblib",
        "*.pickle",
        "*.tar.gz",
        "data/",
        "datasets/",
        "credentials/",
        ".aws/",
    ],
)
def test_dockerignore_excludes_sensitive_or_model_inputs(entry: str) -> None:
    dockerignore = (ROOT / ".dockerignore").read_text().splitlines()

    assert entry in dockerignore


def test_dockerignore_keeps_only_the_metadata_manifest_from_artifacts() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text().splitlines()

    assert "artifacts/*.pkl" in dockerignore
    assert "artifacts/*.joblib" in dockerignore
    assert "artifacts/*.tar.gz" in dockerignore
    assert "!artifacts/*.manifest.json" in dockerignore


def test_smoke_harness_defines_health_only_and_live_prediction_modes() -> None:
    smoke_test = (ROOT / "scripts" / "docker_smoke_test.py").read_text()

    assert "--sample-prediction" in smoke_test
    assert "--model-s3-bucket" in smoke_test
    assert "--model-s3-version-id" in smoke_test
    assert "_stcore/health" in smoke_test
    assert "playwright" in smoke_test
    assert "AWS_ACCESS_KEY_ID" not in smoke_test
    assert "AWS_SECRET_ACCESS_KEY" not in smoke_test
    assert "*.pkl" not in smoke_test


def test_live_prediction_requires_complete_s3_configuration() -> None:
    from scripts.docker_smoke_test import SmokeConfigurationError, validate_live_config

    with pytest.raises(SmokeConfigurationError, match="S3 model configuration"):
        validate_live_config({"IRIS_MODEL_S3_BUCKET": ""})

    with pytest.raises(SmokeConfigurationError, match="S3 model configuration"):
        validate_live_config(
            {
                "IRIS_MODEL_S3_BUCKET": "private-bucket",
                "IRIS_MODEL_S3_KEY": None,
                "IRIS_MODEL_S3_VERSION_ID": "version-1",
            }
        )

    validate_live_config(
        {
            "IRIS_MODEL_S3_BUCKET": "private-bucket",
            "IRIS_MODEL_S3_KEY": "models/iris-v1.tar.gz",
            "IRIS_MODEL_S3_VERSION_ID": "version-1",
        }
    )
