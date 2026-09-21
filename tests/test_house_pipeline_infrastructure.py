"""Static contracts for the House CodePipeline CloudFormation stack."""

from pathlib import Path

ROOT = Path(__file__).parents[1]
TEMPLATE = ROOT / "infra" / "cloudformation" / "house-mlops-pipelines.yml"
PARAMETERS = (
    ROOT / "infra" / "cloudformation" / "parameters" / "house-mlops-production.json"
)


def test_template_defines_v2_pipelines_and_encrypted_artifact_storage() -> None:
    assert TEMPLATE.is_file(), "pipeline CloudFormation template is missing"
    template = TEMPLATE.read_text(encoding="utf-8")

    assert "PipelineType: V2" in template
    assert "house-pricing-model" in template
    assert "house-pricing-app" in template
    assert "AWS::S3::Bucket" in template
    assert "BucketEncryption" in template
    assert "LifecycleConfiguration" in template
    assert "Status: Enabled" in template
    assert template.count("AWS::CodeBuild::Project") >= 4


def test_template_defines_codeconnections_approval_and_pipeline_variables() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")

    assert "CodeStarSourceConnection" in template
    assert "ConnectionArn" in template
    assert "FullRepositoryId" in template
    assert "AWS::SNS::Topic" in template
    assert "Provider: Manual" in template
    assert "NotificationArn" in template
    for variable in (
        "DATASET_VERSION",
        "DATASET_S3_VERSION_ID",
        "MODEL_VERSION",
        "PROMOTION_REASON",
    ):
        assert variable in template


def test_template_uses_separate_scoped_roles_without_static_keys_or_task_wildcards(
) -> None:
    template = TEMPLATE.read_text(encoding="utf-8")

    for role in (
        "HouseCodePipelineRole",
        "HouseModelBuildRole",
        "HouseModelPromotionRole",
        "HouseAppBuildRole",
        "HouseAppDeployRole",
    ):
        assert role in template
    assert "AWS_ACCESS_KEY_ID" not in template
    assert "AWS_SECRET_ACCESS_KEY" not in template
    assert "house-pricing-model-read" in template
    assert "HouseTaskRoleArn" in template
    assert "models/house-price/*" not in template
    assert "models/house-price/v1/model.tar.gz/*" not in template


def test_production_parameter_file_has_no_secret_values() -> None:
    assert PARAMETERS.is_file(), "production parameter file is missing"
    parameters = PARAMETERS.read_text(encoding="utf-8")

    for name in (
        "GitHubConnectionArn",
        "GitHubFullRepositoryId",
        "GitHubBranch",
        "EcsServiceArn",
        "EcsExecutionRoleArn",
        "EcsTaskRoleArn",
        "HouseDatasetS3BucketName",
        "HouseModelS3BucketName",
        "ApprovedReleaseParameterName",
    ):
        assert name in parameters
    assert "AWS_ACCESS_KEY_ID" not in parameters
    assert "AWS_SECRET_ACCESS_KEY" not in parameters
