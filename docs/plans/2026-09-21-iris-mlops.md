# Iris MLOps Showcase Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and deploy a Streamlit Iris classifier that retrieves a verified pre-trained model bundle only from private S3, runs on ECS Express Mode/Fargate, and emits CloudWatch-observable prediction telemetry.

**Architecture:** The repository contains application code, tests, sanitized artifact metadata, infrastructure policy/configuration, and workflows, but no dataset, model bytes, or secrets. A one-time intake process inspects a pinned public educational model, verifies its checksums and schema, and uploads the bundle to a private versioned S3 bucket. ECS tasks download and verify that immutable bundle with a least-privilege task role before caching it for inference. GitHub Actions uses OIDC to push an application-only image to ECR and update ECS Express Mode.

**Tech Stack:** Python 3.12, Streamlit, scikit-learn/joblib compatibility, boto3, pytest, Ruff, Docker, Amazon S3, ECR, ECS Express Mode/Fargate, CloudWatch EMF, IAM, GitHub Actions OIDC.

---

### Task 1: Establish repository scaffolding and safe configuration boundaries

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `.gitignore`
- Create: `.dockerignore`
- Create: `.env.example`
- Create: `src/iris_mlops/__init__.py`
- Create: `src/iris_mlops/config.py`
- Test: `tests/test_config.py`

**Step 1: Write the failing tests**

Test that configuration reads non-secret model location, model version, environment, commit SHA, and image version from environment variables; rejects missing bucket/key/version; and never includes secret values in repr/log-safe output.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -q`

Expected: FAIL because the configuration module does not exist.

**Step 3: Write minimal implementation**

Implement a typed settings object with explicit required variables:

- `IRIS_MODEL_S3_BUCKET`
- `IRIS_MODEL_S3_KEY`
- `IRIS_MODEL_S3_VERSION_ID`
- `IRIS_MODEL_VERSION`
- `IRIS_ENVIRONMENT`
- `GIT_COMMIT_SHA`
- `DOCKER_IMAGE_VERSION`
- `LOW_CONFIDENCE_THRESHOLD`

Keep credentials out of the settings contract. Configure `pyproject.toml` for a `src/` layout, pytest, Ruff, and Python 3.12. Ignore `.env*` except `.env.example`, model archives, datasets, credentials, and local Docker output.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -q`

Expected: PASS.

**Step 5: Commit**

Run: `git add pyproject.toml requirements.txt requirements-dev.txt .gitignore .dockerignore .env.example src tests && git commit -m "build: add safe Iris project scaffolding"`

### Task 2: Define and validate the S3-only artifact manifest

**Files:**
- Create: `artifacts/iris-classifier-v1.manifest.json`
- Create: `src/iris_mlops/manifest.py`
- Test: `tests/test_manifest.py`

**Step 1: Write the failing tests**

Cover required fields, exact ordered feature schema, exact class names, supported framework, semantic model version, source URL/revision, SHA-256 format, Python/scikit-learn metadata, and the rule that training metrics are included only when source-verified. Test that a model manifest never embeds model bytes or dataset rows.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_manifest.py -q`

Expected: FAIL because manifest parsing and validation do not exist.

**Step 3: Write minimal implementation**

Implement strict JSON validation with a `ModelManifest` dataclass or equivalent. The committed manifest contains only sanitized metadata. Initially record the candidate public educational source and a placeholder status until the intake task records the final immutable revision and checksums; do not commit an artifact or dataset.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_manifest.py -q`

Expected: PASS.

**Step 5: Commit**

Run: `git add artifacts/iris-classifier-v1.manifest.json src/iris_mlops/manifest.py tests/test_manifest.py && git commit -m "feat: validate Iris model metadata"`

### Task 3: Implement checksum-verified S3 artifact loading

**Files:**
- Create: `src/iris_mlops/storage.py`
- Create: `src/iris_mlops/model.py`
- Test: `tests/test_model_loading.py`
- Test: `tests/test_schema_validation.py`

**Step 1: Write the failing tests**

Test that the loader:

- downloads the exact S3 key and version ID through a storage abstraction;
- verifies the bundle SHA-256 before deserialization;
- rejects a checksum mismatch;
- rejects path traversal while extracting a bundle;
- validates the four feature names and order;
- validates `setosa`, `versicolor`, `virginica`;
- loads required preprocessing objects when present;
- uses `predict_proba` when supported;
- rejects a probability vector with the wrong shape;
- does not call any fit/train method.

Use mocked S3 and mocked deserialization boundaries in unit tests; do not create or commit a trained model fixture.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model_loading.py tests/test_schema_validation.py -q`

Expected: FAIL because storage and model loading do not exist.

**Step 3: Write minimal implementation**

Implement:

- `S3ArtifactStore.get_versioned_object(bucket, key, version_id)` using boto3.
- in-memory SHA-256 verification;
- safe temporary extraction with an explicit allowlist of bundle filenames;
- manifest validation before joblib loading;
- `LoadedIrisModel.predict(features)` returning class, confidence, and optional probabilities;
- `@st.cache_resource` integration through a cacheable loader function, while keeping the pure loader independently testable.

Use no S3 listing permission and no runtime downloads from public URLs.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model_loading.py tests/test_schema_validation.py -q`

Expected: PASS.

**Step 5: Commit**

Run: `git add src/iris_mlops/storage.py src/iris_mlops/model.py tests/test_model_loading.py tests/test_schema_validation.py && git commit -m "feat: load verified Iris artifacts from S3"`

### Task 4: Add prediction validation and structured CloudWatch logging

**Files:**
- Create: `src/iris_mlops/prediction.py`
- Create: `src/iris_mlops/logging_utils.py`
- Test: `tests/test_prediction.py`
- Test: `tests/test_logging.py`

**Step 1: Write the failing tests**

Test valid prediction, non-finite values, missing values, wrong feature count, non-numeric values, expected class names, probability output shape, confidence threshold behavior, JSON prediction fields, invalid-input events, prediction errors, startup/model-load events, commit SHA/image version fields, and EMF metric dimensions limited to `Environment` and `ModelVersion`.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prediction.py tests/test_logging.py -q`

Expected: FAIL because prediction and logging helpers do not exist.

**Step 3: Write minimal implementation**

Implement schema-based numeric input validation and a prediction service that measures latency with a monotonic clock. Emit one JSON object per event to stdout. Prediction logs must contain `event`, `model_version`, `predicted_class`, `confidence`, `latency_ms`, and `valid_input`. Add EMF blocks for:

- `PredictionCount`
- `PredictionErrorCount`
- `PredictionLatencyMs`
- `LowConfidencePredictionCount`
- `InvalidInputCount`
- `ModelLoadFailure`

Never log raw credentials, model bytes, full S3 URLs containing secrets, or request IDs as dimensions.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prediction.py tests/test_logging.py -q`

Expected: PASS.

**Step 5: Commit**

Run: `git add src/iris_mlops/prediction.py src/iris_mlops/logging_utils.py tests/test_prediction.py tests/test_logging.py && git commit -m "feat: add Iris prediction telemetry"`

### Task 5: Build the Streamlit application

**Files:**
- Create: `app.py`
- Test: `tests/test_app_startup.py`
- Test: `tests/test_app_ui.py`

**Step 1: Write the failing tests**

Test import/startup with a mocked S3 model loader, four numeric controls, invalid-input rejection, prediction button, predicted species, probability display, provenance display, educational text, and no training calls during startup or inference.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_app_startup.py tests/test_app_ui.py -q`

Expected: FAIL because `app.py` does not exist.

**Step 3: Write minimal implementation**

Create the Streamlit interface, call the cached S3 loader once, show a clear model-unavailable state on load failure, and send all prediction/error events through the structured logger. Configure page metadata and avoid exposing bucket credentials or internal exception details to users.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_app_startup.py tests/test_app_ui.py -q`

Expected: PASS.

**Step 5: Commit**

Run: `git add app.py tests/test_app_startup.py tests/test_app_ui.py && git commit -m "feat: add Streamlit Iris application"`

### Task 6: Create the model intake and S3 upload workflow

**Files:**
- Create: `scripts/intake_model.py`
- Create: `scripts/validate_s3_artifact.py`
- Create: `scripts/upload_model_artifact.py`
- Modify: `artifacts/iris-classifier-v1.manifest.json`
- Test: `tests/test_intake_validation.py`

**Step 1: Write the failing tests**

Test that intake requires an explicit source URL/revision and expected checksums, validates downloaded members before packaging, refuses unexpected files, writes only a temporary local bundle, and never writes model bytes into a repository path. Test S3 validation against mocked `GetObject`/`GetObjectVersion` responses.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_intake_validation.py -q`

Expected: FAIL because the intake scripts do not exist.

**Step 3: Write minimal implementation**

Use the inspected public educational artifact source, pinned to an immutable revision. Download to a temporary directory, inspect the model card and serialized members, validate feature/class/probability behavior, calculate checksums, and upload only the final bundle plus manifest to S3. Store the finalized checksums and source metadata in the sanitized manifest. Do not train, convert, or recreate the model.

The upload script must use private S3, server-side encryption, versioning, explicit object keys, and no public ACLs. It must print object keys, version IDs, and checksums only—not credentials or secret values.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_intake_validation.py -q`

Expected: PASS.

**Step 5: Commit**

Run: `git add scripts artifacts/iris-classifier-v1.manifest.json tests/test_intake_validation.py && git commit -m "feat: add verified S3 model intake"`

### Task 7: Containerize the application without model bytes or credentials

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `scripts/docker_smoke_test.py`
- Test: `tests/test_docker_smoke.py`

**Step 1: Write the failing test**

Add a smoke test contract that starts a supplied image with runtime S3 configuration, waits for `/_stcore/health`, confirms port 8501, opens the Streamlit UI, submits representative setosa inputs, and asserts a prediction result. Assert the image filesystem does not contain model artifact paths.

**Step 2: Run the test to verify it fails**

Run: `pytest tests/test_docker_smoke.py -q`

Expected: FAIL because the Docker image and smoke runner do not exist.

**Step 3: Write minimal implementation**

Create a non-root production-oriented image with pinned runtime dependencies, `EXPOSE 8501`, Streamlit bound to `0.0.0.0`, a health check on `/_stcore/health`, and no AWS credential files. Keep model and dataset paths excluded by `.dockerignore`. Use Playwright only for the optional container UI submission check; health checks must remain available with standard HTTP tooling.

**Step 4: Run tests to verify they pass**

Run: `docker build -t iris-mlops:test . && pytest tests/test_docker_smoke.py -q`

Expected: image build succeeds and the smoke test passes when valid S3 runtime configuration is supplied.

**Step 5: Commit**

Run: `git add Dockerfile docker-compose.yml scripts/docker_smoke_test.py tests/test_docker_smoke.py .dockerignore && git commit -m "build: containerize application without model data"`

### Task 8: Add AWS IAM, S3, ECR, ECS Express Mode, and CloudWatch deployment definitions

**Files:**
- Create: `infra/README.md`
- Create: `infra/iam-trust-policies.json`
- Create: `infra/iam-permissions.json`
- Create: `infra/cloudwatch-alarms.json`
- Create: `infra/deploy_express.py`
- Create: `infra/cleanup.py`
- Test: `tests/test_infra_config.py`

**Step 1: Write the failing tests**

Test that infrastructure configuration uses one region, the `iris-mlops` prefix, exact GitHub OIDC subject restriction, no access keys, least-privilege S3 object scope, low-cardinality metric dimensions, and no destructive wildcard cleanup targets.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_infra_config.py -q`

Expected: FAIL because infrastructure definitions do not exist.

**Step 3: Write minimal implementation**

Implement idempotent provisioning/update helpers for:

- private versioned S3 bucket and artifact objects;
- private ECR repository;
- ECS task execution role;
- ECS Express infrastructure role;
- ECS task role limited to the model S3 prefix;
- CloudWatch log group and retention;
- ECS Express Mode service using the application-only image and port 8501;
- CloudWatch alarms for EMF application metrics and discovered load-balancer/service health;
- GitHub OIDC deployment role restricted to `repo:DJaegerScript/MLOps-iris-house:ref:refs/heads/main`.

Before any mutation, the deployment procedure must verify account `163918295215`, region `ap-southeast-3`, existing `iris-mlops` resources, and the billing budget state. It must never delete or modify unrelated resources. Cleanup accepts only explicit project resource names/ARNs.

Use the AWS integration for the actual account provisioning. Record resource names, ARNs, S3 object version ID, and Express URL in a local deployment report that is ignored by Git.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_infra_config.py -q`

Expected: PASS.

**Step 5: Commit**

Run: `git add infra tests/test_infra_config.py && git commit -m "feat: define least-privilege AWS deployment"`

### Task 9: Add GitHub Actions workflows

**Files:**
- Create: `.github/workflows/pull-request.yml`
- Create: `.github/workflows/deploy.yml`
- Test: `tests/test_workflows.py`

**Step 1: Write the failing tests**

Test workflow YAML for pinned/setup dependencies, tests, Ruff, Docker build, OIDC permissions, SHA image tags, ECR push, ECS Express update, model version reporting, and absence of training commands, static AWS keys, or secret values.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_workflows.py -q`

Expected: FAIL because workflow files do not exist.

**Step 3: Write minimal implementation**

Create PR and deployment workflows. Deployment uses `id-token: write`, `aws-actions/configure-aws-credentials`, ECR login, an image tag based on `GITHUB_SHA`, and the pre-provisioned OIDC role. Pass only non-secret S3 bucket/key/version/model-version configuration to ECS. Do not upload model bytes from GitHub and do not add a training job.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_workflows.py -q`

Expected: PASS.

**Step 5: Commit**

Run: `git add .github tests/test_workflows.py && git commit -m "ci: add Iris validation and OIDC deployment workflows"`

### Task 10: Write the operational README and run the full local verification loop

**Files:**
- Create: `README.md`
- Create: `docs/deployment-report.template.md`
- Test: `tests/test_readme_contract.py`

**Step 1: Write the failing test**

Test that the README includes purpose, architecture, pre-trained model rationale, included/excluded MLOps stages, local setup, Streamlit, Docker, AWS prerequisites, deployment, CloudWatch logs/metrics, provenance, limitations, rollback, cleanup, cost scope, and explicit no-training statements.

**Step 2: Run the test to verify it fails**

Run: `pytest tests/test_readme_contract.py -q`

Expected: FAIL because the README does not exist.

**Step 3: Write minimal implementation**

Document the S3-only model boundary, how to perform intake, required runtime variables, local mock/integration test modes, Docker execution, AWS setup, GitHub OIDC, CloudWatch queries, rollback to a previous ECR image/model S3 version, and cleanup commands. State that no dataset, model bytes, or secrets are committed.

**Step 4: Run all focused checks**

Run: `ruff check . && pytest -q`

Expected: no lint errors and all tests pass.

**Step 5: Commit**

Run: `git add README.md docs/deployment-report.template.md tests/test_readme_contract.py && git commit -m "docs: document Iris MLOps operations"`

### Task 11: Provision, deploy, and perform live verification

**Files:**
- Modify: `docs/deployment-report.md` (ignored or generated outside Git if it contains account-specific operational data)

**Step 1: Verify account and existing resources**

Use the connected AWS integration to re-check account ID, region, prefix collisions, unrelated resources, and billing budget state. If a budget alert exists, record its non-secret configuration. If no notification address is available, request one before creating the alert.

**Step 2: Upload the verified model to S3**

Run the intake/upload workflow in an isolated temporary location, validate the real bundle, upload it with versioning/encryption, and record the object key, version ID, and checksum without committing the bundle.

**Step 3: Provision AWS resources**

Create only the named S3, ECR, IAM, ECS Express, CloudWatch, and budget resources. Wait for IAM propagation and Express Mode deployment completion.

**Step 4: Build and push the application image**

Build the application-only image, tag it with the Git commit SHA, scan/report the image metadata, and push it to the private ECR repository.

**Step 5: Deploy and verify**

Update ECS Express Mode, poll until active, then verify the public URL, `/_stcore/health`, a live setosa prediction, structured CloudWatch logs, custom EMF metrics, application alarms, and service-health alarms.

**Step 6: Record and review**

Record resource names/ARNs, public URL, image tag/digest, model version/S3 version ID, and verification timestamps. Run `git status --short --branch` and confirm no model, dataset, secret, or deployment report containing sensitive values is tracked.

**Step 7: Commit only safe metadata**

Commit sanitized documentation/report metadata only. Never commit credentials, model bytes, dataset files, account secrets, or runtime secret values.

