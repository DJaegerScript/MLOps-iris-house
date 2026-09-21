"""Static contracts for the GitHub Actions workflows."""

from __future__ import annotations

from pathlib import Path

WORKFLOW_DIR = Path(__file__).parents[1] / ".github" / "workflows"


def _workflow(name: str) -> str:
    return (WORKFLOW_DIR / name).read_text(encoding="utf-8")


def test_pull_request_workflow_runs_checks_and_builds_only_the_application_image(
) -> None:
    workflow = _workflow("pull-request.yml")

    assert "pull_request:" in workflow
    assert "contents: read" in workflow
    assert "setup-python@" in workflow
    assert (
        "python -m pip install -r requirements.txt -r requirements-dev.txt"
        in workflow
    )
    assert "python -m pytest -q" in workflow
    assert "python -m ruff check ." in workflow
    assert "docker build" in workflow
    assert "Dockerfile" in workflow
    assert "id-token: write" not in workflow
    assert "aws-actions/configure-aws-credentials" not in workflow
    assert "train" not in workflow.lower()
    assert "dataset" not in workflow.lower()
    assert "model.pkl" not in workflow.lower()


def test_deployment_workflow_is_main_only_oidc_and_sha_tagged() -> None:
    workflow = _workflow("deploy.yml")

    assert "push:" in workflow
    assert "branches: [main]" in workflow
    assert "workflow_dispatch:" in workflow
    assert "id-token: write" in workflow
    assert "contents: read" in workflow
    assert "aws-actions/configure-aws-credentials@" in workflow
    assert "role-to-assume: ${{ env.AWS_DEPLOY_ROLE_ARN }}" in workflow
    assert (
        "repo:DJaegerScript@85334514/MLOps-iris-house@1378975661:ref:refs/heads/main"
        in workflow
    )
    assert "github.sha" in workflow
    assert "environment: production" not in workflow
    assert "image_exists" in workflow
    assert "describe-images" in workflow
    assert "docker push" in workflow
    assert "create-express-gateway-service" in workflow
    assert "update-express-gateway-service" in workflow
    assert "list-service-deployments" in workflow
    assert "describe-service-deployments" in workflow
    assert "monitor-express-gateway-service" not in workflow
    assert "/_stcore/health" in workflow
    assert "IRIS_MODEL_S3_BUCKET" in workflow
    assert "IRIS_MODEL_S3_KEY" in workflow
    assert "IRIS_MODEL_S3_VERSION_ID" in workflow
    assert "IRIS_MODEL_VERSION" in workflow
    assert "GIT_COMMIT_SHA" in workflow
    assert "DOCKER_IMAGE_VERSION" in workflow
    assert "AWS_ACCESS_KEY_ID" not in workflow
    assert "AWS_SECRET_ACCESS_KEY" not in workflow
    assert "dataset" not in workflow.lower()
    assert "model.pkl" not in workflow.lower()
    assert "train" not in workflow.lower()
