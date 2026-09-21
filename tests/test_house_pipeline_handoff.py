"""Tests for the approved House release to application-pipeline handoff."""

import importlib
import json
from pathlib import Path

import pytest

from house_pricing_mlops.release import HouseReleaseManifest, ReleaseValidationError

HANDOFF_MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "codepipeline"
    / "invoke_house_app_pipeline.py"
)


def _handoff_api():
    assert HANDOFF_MODULE_PATH.is_file(), "pipeline handoff implementation is missing"
    return importlib.import_module("scripts.codepipeline.invoke_house_app_pipeline")


def _manifest(*, status: str = "champion") -> HouseReleaseManifest:
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
            "status": status,
            "approved_by": "codepipeline/automatic" if status == "champion" else None,
            "approval_reason": "automatic promotion" if status == "champion" else None,
        }
    )


class FakeCodePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def start_pipeline_execution(self, **kwargs: object) -> dict[str, str]:
        self.calls.append(kwargs)
        return {"pipelineExecutionId": "execution-123"}


class FakeSsm:
    def __init__(self, value: str) -> None:
        self.value = value

    def get_parameter(self, **kwargs: object) -> dict[str, object]:
        assert kwargs == {"Name": "/approved/house", "WithDecryption": False}
        return {"Parameter": {"Value": self.value}}


def test_promoted_release_contains_exact_model_version_id() -> None:
    release = _manifest()

    assert release.status == "champion"
    assert release.model_s3_version_id == "model-version-2"
    assert release.model_s3_version_id != "latest"


def test_handoff_starts_app_pipeline_with_exact_release_coordinates() -> None:
    client = FakeCodePipeline()
    handoff_api = _handoff_api()

    execution_id = handoff_api.start_app_pipeline(
        client,
        pipeline_name="house-pricing-app",
        release=_manifest(),
    )

    assert execution_id == "execution-123"
    assert client.calls[0]["name"] == "house-pricing-app"
    variables = {
        item["name"]: item["value"] for item in client.calls[0]["variables"]
    }
    assert variables["HOUSE_MODEL_S3_BUCKET"] == "private-bucket"
    assert variables["HOUSE_MODEL_S3_KEY"] == "models/house-price/v2/model.tar.gz"
    assert variables["HOUSE_MODEL_S3_VERSION_ID"] == "model-version-2"
    assert variables["HOUSE_MODEL_VERSION"] == "v2"
    assert variables["HOUSE_DATASET_VERSION"] == "v3"
    assert "latest" not in json.dumps(variables).lower()


def test_handoff_rejects_candidate_or_incomplete_release(tmp_path: Path) -> None:
    handoff_api = _handoff_api()
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps({**_manifest().to_dict(), "status": "candidate"}),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseValidationError, match="champion"):
        handoff_api.load_approved_release(candidate_path)

    incomplete = _manifest().to_dict()
    incomplete["model_s3_version_id"] = ""
    with pytest.raises(ReleaseValidationError, match="model_s3_version_id"):
        HouseReleaseManifest.from_dict(incomplete)


def test_code_only_deployment_resolves_current_approved_ssm_release() -> None:
    handoff_api = _handoff_api()
    release = handoff_api.resolve_release_from_ssm(
        FakeSsm(json.dumps(_manifest().to_dict())),
        parameter_name="/approved/house",
    )

    assert release.status == "champion"
    assert release.model_s3_version_id == "model-version-2"


def test_buildspecs_persist_and_consume_the_approved_release_pointer() -> None:
    root = Path(__file__).parents[1]
    promotion = (root / "buildspecs" / "house-model-promote.yml").read_text()
    deployment = (root / "buildspecs" / "house-app-deploy.yml").read_text()

    assert "aws ssm put-parameter" in promotion
    assert "APPROVED_RELEASE_PARAMETER_NAME" in promotion
    assert "invoke_house_app_pipeline.py" in promotion
    assert "APP_PIPELINE_NAME" in promotion
    assert "aws ssm get-parameter" in deployment
    assert "approved-release.json" in deployment
    assert "HOUSE_MODEL_S3_VERSION_ID" in deployment
