"""Tests for safe Iris application configuration."""

from dataclasses import fields

import pytest

from iris_mlops.config import Settings, load_config

REQUIRED_ENVIRONMENT = {
    "IRIS_MODEL_S3_BUCKET": "iris-models-example",
    "IRIS_MODEL_S3_KEY": "models/iris-classifier/v1/bundle.tar.gz",
    "IRIS_MODEL_S3_VERSION_ID": "s3-version-id",
    "IRIS_MODEL_VERSION": "v2",
    "IRIS_ENVIRONMENT": "staging",
    "GIT_COMMIT_SHA": "abc123",
    "DOCKER_IMAGE_VERSION": "abc123",
    "LOW_CONFIDENCE_THRESHOLD": "0.65",
}


def _set_required_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable, value in REQUIRED_ENVIRONMENT.items():
        monkeypatch.setenv(variable, value)


def test_load_config_reads_all_required_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_environment(monkeypatch)

    settings = load_config()

    assert settings == Settings(
        model_s3_bucket="iris-models-example",
        model_s3_key="models/iris-classifier/v1/bundle.tar.gz",
        model_s3_version_id="s3-version-id",
        model_version="v2",
        environment="staging",
        git_commit_sha="abc123",
        docker_image_version="abc123",
        low_confidence_threshold=0.65,
    )


@pytest.mark.parametrize("missing_variable", sorted(REQUIRED_ENVIRONMENT))
def test_load_config_rejects_missing_required_values(
    monkeypatch: pytest.MonkeyPatch,
    missing_variable: str,
) -> None:
    _set_required_environment(monkeypatch)
    monkeypatch.delenv(missing_variable)

    with pytest.raises(ValueError, match=missing_variable):
        load_config()


@pytest.mark.parametrize("blank_variable", sorted(REQUIRED_ENVIRONMENT))
def test_load_config_rejects_blank_required_values(
    monkeypatch: pytest.MonkeyPatch,
    blank_variable: str,
) -> None:
    _set_required_environment(monkeypatch)
    monkeypatch.setenv(blank_variable, "  ")

    with pytest.raises(ValueError, match=blank_variable):
        load_config()


@pytest.mark.parametrize("threshold", ["not-a-number", "-0.01", "1.01"])
def test_load_config_rejects_invalid_confidence_threshold(
    monkeypatch: pytest.MonkeyPatch,
    threshold: str,
) -> None:
    _set_required_environment(monkeypatch)
    monkeypatch.setenv("LOW_CONFIDENCE_THRESHOLD", threshold)

    with pytest.raises(ValueError, match="LOW_CONFIDENCE_THRESHOLD"):
        load_config()


def test_settings_expose_only_non_secret_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_environment(monkeypatch)
    secret = "do-not-log-this-secret"
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret)
    monkeypatch.setenv("AWS_SESSION_TOKEN", secret)

    settings = load_config()

    assert secret not in repr(settings)
    assert secret not in str(settings)
    assert [field.name for field in fields(settings)] == [
        "model_s3_bucket",
        "model_s3_key",
        "model_s3_version_id",
        "model_version",
        "environment",
        "git_commit_sha",
        "docker_image_version",
        "low_confidence_threshold",
    ]
