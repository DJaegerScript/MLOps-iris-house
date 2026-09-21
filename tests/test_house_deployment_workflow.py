"""Static deployment and IAM contracts for the shared ECS application."""

from pathlib import Path

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "deploy.yml"


def test_deployment_keeps_iris_coordinates_and_adds_exact_house_coordinates() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    for name in (
        "IRIS_MODEL_S3_BUCKET",
        "IRIS_MODEL_S3_KEY",
        "IRIS_MODEL_S3_VERSION_ID",
        "IRIS_MODEL_VERSION",
        "HOUSE_MODEL_S3_BUCKET",
        "HOUSE_MODEL_S3_KEY",
        "HOUSE_MODEL_S3_VERSION_ID",
        "HOUSE_MODEL_VERSION",
        "HOUSE_DATASET_VERSION",
    ):
        assert name in workflow
    assert "github.sha" in workflow
    assert "/_stcore/health" in workflow
    assert '"name": "HOUSE_MODEL_S3_VERSION_ID"' in workflow
    assert '"name": "IRIS_MODEL_S3_VERSION_ID"' in workflow


def test_deployment_reports_both_products_and_uses_oidc_without_static_keys() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "### Iris deployment" in workflow
    assert "### House Pricing deployment" in workflow
    assert "aws-actions/configure-aws-credentials@" in workflow
    assert "AWS_ACCESS_KEY_ID" not in workflow
    assert "AWS_SECRET_ACCESS_KEY" not in workflow
    assert "model.tar.gz" not in workflow.lower()


def test_deployment_policy_check_is_exact_house_model_read_scope() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "get-role-policy" in workflow
    assert "house-pricing-model-read" in workflow
    assert '"s3:GetObject"' in workflow
    assert '"s3:GetObjectVersion"' in workflow
    assert "HOUSE_MODEL_S3_KEY" in workflow
    assert "/*" not in workflow


def test_aws_house_guide_names_shared_resources_and_rollback_contract() -> None:
    guide = (ROOT / "docs" / "aws-house-pricing.md").read_text(encoding="utf-8")

    assert "ap-southeast-3" in guide
    assert "iris-mlops-models-163918295215-apse3" in guide
    assert "/aws/ecs/iris-mlops" in guide
    assert "models/house-price/" in guide
    assert "VersionId" in guide
    assert "rollback" in guide.lower()
    assert "AWS_ACCESS_KEY_ID" not in guide
