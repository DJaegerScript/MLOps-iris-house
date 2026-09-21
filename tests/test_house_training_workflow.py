"""Static contracts for the manual House Pricing training workflow."""

from pathlib import Path

WORKFLOW = (
    Path(__file__).parents[1] / ".github" / "workflows" / "train-house-price.yml"
)


def test_training_workflow_is_manual_or_narrowly_scoped_and_uses_oidc() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "paths:" in workflow
    assert "src/house_pricing_mlops/**" in workflow
    assert "scripts/train_house_price_model.py" in workflow
    assert "id-token: write" in workflow
    assert "aws-actions/configure-aws-credentials@" in workflow
    assert "HOUSE_TRAINING_ROLE_ARN" in workflow
    assert "AWS_ACCESS_KEY_ID" not in workflow
    assert "AWS_SECRET_ACCESS_KEY" not in workflow


def test_training_workflow_uses_exact_dataset_and_publishes_bundle_metadata() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "HOUSE_DATASET_S3_BUCKET" in workflow
    assert "HOUSE_DATASET_S3_KEY" in workflow
    assert "HOUSE_DATASET_S3_VERSION_ID" in workflow
    assert "--version-id" in workflow
    assert "requirements-training.txt" in workflow
    assert "--random-seed" in workflow
    assert "MLFLOW_TRACKING_URI" in workflow
    assert "model.tar.gz" in workflow
    assert "s3api put-object" in workflow
    assert "register-candidate" in workflow
    assert "GITHUB_STEP_SUMMARY" in workflow
    assert "MLflow run" in workflow


def test_production_promotion_is_explicit_and_not_deployment() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "promote" in workflow
    assert "environment: house-pricing-production" in workflow
    assert "promotion_reason" in workflow
    assert "promote_champion" in workflow
    assert "deploy.yml" not in workflow
    assert "update-express-gateway-service" not in workflow
