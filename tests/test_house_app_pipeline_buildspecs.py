"""Static contracts for the shared application CodeBuild stages."""

from pathlib import Path

ROOT = Path(__file__).parents[1]
BUILD_BUILDSPEC = ROOT / "buildspecs" / "house-app-build.yml"
DEPLOY_BUILDSPEC = ROOT / "buildspecs" / "house-app-deploy.yml"
WAIT_HELPER = ROOT / "scripts" / "codepipeline" / "wait_for_ecs_express_deployment.sh"


def test_application_buildspec_runs_checks_and_publishes_digest_metadata() -> None:
    assert BUILD_BUILDSPEC.is_file(), "application buildspec is missing"
    buildspec = BUILD_BUILDSPEC.read_text(encoding="utf-8")

    assert "requirements.txt" in buildspec
    assert "requirements-dev.txt" in buildspec
    assert "requirements-training.txt" in buildspec
    assert "pytest -q" in buildspec
    assert "ruff check ." in buildspec
    assert "CODEBUILD_RESOLVED_SOURCE_VERSION" in buildspec
    assert "get-login-password" in buildspec
    assert "docker build" in buildspec
    assert "docker push" in buildspec
    assert "describe-images" in buildspec
    assert "image-release.json" in buildspec
    assert "image_uri" in buildspec
    assert "commit_sha" in buildspec
    assert "digest" in buildspec
    assert "model.tar.gz" not in buildspec
    assert "datasets/" not in buildspec


def test_deployment_buildspec_uses_exact_releases_and_house_ecs_express() -> None:
    assert DEPLOY_BUILDSPEC.is_file(), "deployment buildspec is missing"
    buildspec = DEPLOY_BUILDSPEC.read_text(encoding="utf-8")

    for name in (
        "image-release.json",
        "approved-release.json",
        "APP_VARIANT",
        "HOUSE_MODEL_S3_BUCKET",
        "HOUSE_MODEL_S3_KEY",
        "HOUSE_MODEL_S3_VERSION_ID",
        "HOUSE_MODEL_VERSION",
        "HOUSE_DATASET_VERSION",
        "ECS_SERVICE_ARN",
        "ECS_SERVICE_NAME",
        "ECS_CLUSTER_ARN",
        "ECS_INFRASTRUCTURE_ROLE_ARN",
    ):
        assert name in buildspec
    assert "house-pricing-model-read" in buildspec
    assert '"s3:GetObject"' in buildspec
    assert '"s3:GetObjectVersion"' in buildspec
    assert '"containerPort": 8501' in buildspec
    assert "aws ecs create-express-gateway-service" in buildspec
    assert "aws ecs update-express-gateway-service" in buildspec
    assert '"logStreamPrefix": "house-pricing"' in buildspec
    assert '"name": "APP_VARIANT", "value": "house"' in buildspec
    assert '"/_stcore/health"' in buildspec
    assert "wait_for_ecs_express_deployment.sh" in buildspec
    assert "SUCCESSFUL" in buildspec
    assert "curl" in buildspec
    assert "imagedefinitions.json" not in buildspec
    assert "iris-mlops/iris-mlops" not in buildspec


def test_wait_helper_distinguishes_new_success_and_failure_states() -> None:
    assert WAIT_HELPER.is_file(), "ECS Express wait helper is missing"
    helper = WAIT_HELPER.read_text(encoding="utf-8")

    assert "list-service-deployments" in helper
    assert "previous_deployment" in helper
    assert "describe-service-deployments" in helper
    assert "SUCCESSFUL" in helper
    assert "FAILED" in helper
    assert "STOPPED" in helper
    assert "timeout" in helper.lower()
