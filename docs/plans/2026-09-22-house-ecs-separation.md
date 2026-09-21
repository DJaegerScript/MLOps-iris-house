# Separate House ECS Deployment Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep the existing Iris ECS service untouched while deploying House Pricing from CodePipeline to a separate House-only ECS Express service.

**Architecture:** The repository and ECR image remain shared, but runtime behavior is selected by `APP_VARIANT`. The existing `iris-mlops` service keeps its current default Iris behavior. CodePipeline creates or updates a separate `house-pricing` ECS Express service with a dedicated task role, log group, endpoint, and `APP_VARIANT=house`.

**Tech Stack:** Python 3.12, Streamlit, pytest, Ruff, AWS CloudFormation, CodePipeline V2, CodeBuild, ECS Express Gateway, IAM, ECR, S3, SSM, CloudWatch Logs.

---

### Task 1: Add failing runtime-variant tests

**Files:**
- Modify: `tests/test_house_app.py`
- Modify: `tests/test_app.py` if the existing app tests are split there

**Step 1: Write the failing tests**

Add page-boundary tests proving:

- the default runtime renders the Iris product selector/current behavior;
- `APP_VARIANT=house` renders House Pricing without rendering the product selector or invoking Iris configuration/model loading;
- an unsupported variant fails safely rather than silently deploying the wrong product.

**Step 2: Run the focused tests**

Run: `.venv/bin/python -m pytest tests/test_house_app.py -q`

Expected: FAIL because runtime variant selection does not exist.

**Step 3: Commit the failing tests**

```bash
git add tests/test_house_app.py tests/test_app.py
git commit -m "test: define separate House runtime boundary"
```

### Task 2: Implement the House-only application variant

**Files:**
- Modify: `app.py`
- Modify: `tests/test_house_app.py`

**Step 1: Implement the minimal variant boundary**

Read `APP_VARIANT` with default `iris`. Keep the current Iris path unchanged for the default. For `house`, set the page configuration and render only `_render_house_page`; do not call `_selected_product`, `load_config`, or `load_runtime_service`. Reject unsupported values with a safe Streamlit error.

**Step 2: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_house_app.py -q`

Expected: PASS.

**Step 3: Run the existing app tests**

Run: `.venv/bin/python -m pytest tests/test_house_app.py tests/test_workflows.py -q`

Expected: PASS with the default Iris behavior preserved.

**Step 4: Commit**

```bash
git add app.py tests/test_house_app.py
git commit -m "feat: isolate House runtime variant"
```

### Task 3: Define separate House ECS configuration and IAM

**Files:**
- Modify: `infra/cloudformation/house-mlops-pipelines.yml`
- Modify: `infra/cloudformation/parameters/house-mlops-production.json`
- Modify: `tests/test_house_pipeline_infrastructure.py`

**Step 1: Add infrastructure contract tests**

Assert the template contains:

- a separate House service ARN/name and cluster configuration;
- a dedicated House task role and exact model-object `GetObject`/`GetObjectVersion` policy;
- a separate House CloudWatch log group;
- no use of the existing `iris-mlops/iris-mlops` service ARN in House deployment environment variables;
- House runtime environment `APP_VARIANT=house`.

**Step 2: Run the tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_house_pipeline_infrastructure.py -q`

Expected: FAIL because the template still targets the Iris service.

**Step 3: Implement the infrastructure contract**

Add parameters for `HouseEcsServiceArn`, `HouseEcsServiceName`, `HouseEcsClusterArn`, `HouseEcsTaskRoleArn`, `HouseCloudWatchLogGroupName`, and `EcsInfrastructureRoleArn`. Add a dedicated `HouseEcsTaskRole` with the exact versioned House model S3 object policy and a retained House log group. Reuse the existing execution and ECS infrastructure roles. Keep all existing Iris parameter values and resources unchanged.

**Step 4: Run focused validation**

Run:

```bash
.venv/bin/python -m pytest tests/test_house_pipeline_infrastructure.py -q
aws cloudformation validate-template --template-body file://infra/cloudformation/house-mlops-pipelines.yml --region ap-southeast-3
```

Expected: PASS.

**Step 5: Commit**

```bash
git add infra/cloudformation/house-mlops-pipelines.yml infra/cloudformation/parameters/house-mlops-production.json tests/test_house_pipeline_infrastructure.py
git commit -m "feat: define isolated House ECS service contract"
```

### Task 4: Make the House deploy stage create or update only House ECS

**Files:**
- Modify: `buildspecs/house-app-deploy.yml`
- Modify: `tests/test_house_app_pipeline_buildspecs.py`
- Modify: `scripts/codepipeline/rollback_house_release.py` if the service target is hard-coded

**Step 1: Add failing buildspec tests**

Assert the deploy buildspec:

- uses `APP_VARIANT=house`;
- invokes `create-express-gateway-service` when the House ARN is absent;
- invokes `update-express-gateway-service` only with the House service ARN;
- uses the House log group and House task role;
- waits for the House deployment and health URL;
- contains no `iris-mlops/iris-mlops` target.

**Step 2: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_house_app_pipeline_buildspecs.py -q`

Expected: FAIL until the deploy contract is updated.

**Step 3: Implement create/update behavior**

Build the immutable primary container with `APP_VARIANT=house`, exact approved model coordinates, House task role, and House log configuration. Probe the configured House service ARN. If it does not exist, call `aws ecs create-express-gateway-service` with the configured cluster, execution role, infrastructure role, service name, primary container, port 8501, and `/_stcore/health`; otherwise call `update-express-gateway-service`. Capture the service deployment ARN and endpoint, wait for successful completion, and write a House deployment summary.

**Step 4: Update rollback targeting**

Ensure rollback passes the House service variables and never defaults to the existing Iris service.

**Step 5: Run focused tests and commit**

```bash
.venv/bin/python -m pytest tests/test_house_app_pipeline_buildspecs.py tests/test_house_pipeline_operations.py -q
git add buildspecs/house-app-deploy.yml scripts/codepipeline/rollback_house_release.py tests/test_house_app_pipeline_buildspecs.py
git commit -m "feat: deploy House through isolated ECS service"
```

### Task 5: Point CodePipeline roles and variables at House-only deployment

**Files:**
- Modify: `infra/cloudformation/house-mlops-pipelines.yml`
- Modify: `scripts/codepipeline/invoke_house_app_pipeline.py`
- Modify: `tests/test_house_pipeline_handoff.py`
- Modify: `tests/test_house_pipeline_infrastructure.py`

**Step 1: Add failing role/variable assertions**

Assert the app deploy role can pass only the dedicated House task role and update/create the configured House service, and the promotion handoff includes the House service target without changing the Iris pipeline.

**Step 2: Implement scoped permissions and variables**

Replace the shared `EcsServiceArn` deployment reference with House-specific values in the House app deploy project. Grant only the required `ecs:CreateExpressGatewayService`, `ecs:DescribeExpressGatewayService`, `ecs:DescribeServiceDeployments`, `ecs:ListServiceDeployments`, and `ecs:UpdateExpressGatewayService` permissions, plus pass-role access to the existing execution role, dedicated House task role, and infrastructure role. Keep the existing Iris service ARN out of this CodePipeline path.

**Step 3: Run tests and commit**

```bash
.venv/bin/python -m pytest tests/test_house_pipeline_handoff.py tests/test_house_pipeline_infrastructure.py -q
git add infra/cloudformation/house-mlops-pipelines.yml scripts/codepipeline/invoke_house_app_pipeline.py tests/test_house_pipeline_handoff.py tests/test_house_pipeline_infrastructure.py
git commit -m "fix: scope House pipeline to House ECS service"
```

### Task 6: Update House smoke checks and documentation

**Files:**
- Modify: `scripts/codepipeline/smoke_test_house_release.py`
- Modify: `docs/aws-house-pricing.md`
- Modify: `tests/test_house_deployment_workflow.py`
- Modify: `.github/workflows/train-house-price.yml` if it documents the old shared target

**Step 1: Add/update tests**

Assert smoke/configuration documentation uses the House endpoint and documents that Iris remains on its existing endpoint/service.

**Step 2: Implement documentation and smoke target changes**

Remove shared-service wording, document the separate House endpoint and task role, and make rollback/smoke instructions explicit about not touching `iris-mlops`.

**Step 3: Run checks and commit**

```bash
.venv/bin/python -m pytest tests/test_house_deployment_workflow.py tests/test_house_pipeline_operations.py -q
git add scripts/codepipeline/smoke_test_house_release.py docs/aws-house-pricing.md tests/test_house_deployment_workflow.py .github/workflows/train-house-price.yml
git commit -m "docs: record separate Iris and House deployments"
```

### Task 7: Full local verification before AWS mutation

Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
aws cloudformation validate-template --template-body file://infra/cloudformation/house-mlops-pipelines.yml --region ap-southeast-3
git diff --check
git status --short --branch
```

Expected: all tests pass, Ruff is clean, CloudFormation validates, and only intended commits/files are present.

### Task 8: Deploy isolated House infrastructure and verify Iris preservation

Before mutation, record the existing Iris service ARN, endpoint, image digest/tag, primary container environment, task role, and deployment status. Deploy the updated CloudFormation stack in `ap-southeast-3`, create the House service through the House pipeline, and wait for its ECS Express deployment to succeed.

Verify:

- `iris-mlops/iris-mlops` ARN, endpoint, image, environment, task role, and deployment status are unchanged;
- `house-pricing-app` targets the House service ARN;
- the House endpoint health check succeeds;
- the House container has `APP_VARIANT=house`, the promoted image digest, and the champion model VersionId;
- House prediction smoke succeeds;
- no Iris deployment was started by the House pipeline.

Commit any documentation-only live endpoint updates separately and leave the worktree clean.
