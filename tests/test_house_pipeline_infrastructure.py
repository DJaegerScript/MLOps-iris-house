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


def test_template_defines_github_webhook_approval_and_pipeline_variables() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")

    assert "Provider: GitHub" in template
    assert "OAuthToken" in template
    assert "AWS::CodePipeline::Webhook" in template
    assert "GITHUB_HMAC" in template
    assert "admin:repo_hook" in template
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


def test_template_defines_isolated_house_ecs_target_and_runtime_contract() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")

    for name in (
        "HouseEcsServiceArn",
        "HouseEcsServiceName",
        "HouseEcsClusterArn",
        "HouseEcsTaskRoleArn",
        "HouseCloudWatchLogGroupName",
        "EcsInfrastructureRoleArn",
        "HouseEcsTaskRole",
        "HouseCloudWatchLogGroup",
    ):
        assert name in template
    assert "house-pricing" in template
    assert "/aws/ecs/house-pricing" in template
    assert "APP_VARIANT" in template

    deploy_project = template[template.index("HouseAppDeployProject") :]
    assert "Ref: EcsServiceArn" not in deploy_project
    assert "Ref: HouseEcsServiceArn" in deploy_project
    assert "Ref: HouseEcsTaskRoleArn" in deploy_project


def test_house_deploy_role_is_scoped_to_house_service_roles() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    role_start = template.index("  HouseAppDeployRole:")
    role_end = template.index("  HouseModelTrainProject:", role_start)
    deploy_role = template[role_start:role_end]

    for action in (
        "ecs:CreateExpressGatewayService",
        "ecs:DescribeExpressGatewayService",
        "ecs:DescribeServiceDeployments",
        "ecs:ListServiceDeployments",
        "ecs:UpdateExpressGatewayService",
    ):
        assert action in deploy_role
    assert "Ref: EcsExecutionRoleArn" in deploy_role
    assert "Ref: EcsInfrastructureRoleArn" in deploy_role
    assert "Ref: HouseEcsTaskRoleArn" in deploy_role
    assert (
        "arn:aws:ecs:ap-southeast-3:163918295215:service/iris-mlops/iris-mlops"
        not in deploy_role
    )

    project_start = template.index("  HouseAppDeployProject:")
    project_end = template.index("  HouseModelPipeline:", project_start)
    deploy_project = template[project_start:project_end]
    assert "Ref: HouseEcsServiceArn" in deploy_project
    assert "Ref: EcsServiceArn" not in deploy_project


def test_production_parameter_file_has_no_secret_values() -> None:
    assert PARAMETERS.is_file(), "production parameter file is missing"
    parameters = PARAMETERS.read_text(encoding="utf-8")

    for name in (
        "GitHubOAuthTokenSecretArn",
        "GitHubWebhookSecretName",
        "GitHubOwner",
        "GitHubRepository",
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
