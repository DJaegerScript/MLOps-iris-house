# House Pricing MLOps Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a fully testable House Pricing training-to-production lifecycle as a second, lazily loaded Streamlit product while preserving the existing Iris artifact, deployment, and CloudWatch behavior.

**Architecture:** Keep one Streamlit/ECS service. Isolate House Pricing into `src/house_pricing_mlops/`, use a complete fitted scikit-learn pipeline in an immutable S3 bundle, and share only stable infrastructure boundaries with Iris. Use a committed fixture for tests and the real Kaggle `train.csv` only through an external intake step.

**Tech Stack:** Python 3.11+, Streamlit, pandas, NumPy, scikit-learn, joblib, MLflow, boto3, pytest, Ruff, Docker, GitHub Actions, S3 versioning, ECS Express Mode, CloudWatch JSON/EMF, and SageMaker AI MLflow Tracking Server when configured.

---

## Execution rules

- Work only on `feat/house-pricing-mlops`, which starts at the clean approved `main` commit.
- Preserve unrelated files and all existing Iris contracts.
- Use TDD for behavior: write one focused failing test, run it, implement the smallest passing behavior, rerun focused and relevant regression tests, then refactor.
- Never put Kaggle credentials, `train.csv`, generated model bundles, MLflow runs, or downloaded archives in Git.
- Use the committed fixture only for tests; never describe fixture metrics as production results.
- Keep AWS changes after local code, tests, and documentation are verified. Use the existing least-privilege OIDC role for deployment mutations.
- When the real Kaggle dataset is unavailable, stop at the documented intake boundary and report that production training/deployment is pending; do not fabricate results.

## Task 1: Add dependency and repository boundaries

**Files:**

- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `requirements-dev.txt`
- Create: `requirements-training.txt`
- Modify: `.gitignore`
- Modify: `.dockerignore`
- Create: `src/house_pricing_mlops/__init__.py`
- Create: `tests/fixtures/house_prices/train.csv`
- Create: `tests/test_house_package.py`

### Step 1: Write the failing test

Add tests proving the package imports, the fixture is present, the fixture is
not ignored by Git, and runtime requirements do not include credentials or
model/data files.

### Step 2: Run to verify failure

```bash
python3 -m pytest tests/test_house_package.py -q
```

Expected: FAIL because the package and fixture do not exist.

### Step 3: Add the minimal boundary

Add pandas to the runtime dependencies. Add MLflow and any training-only
dependencies to `requirements-training.txt`, which includes the runtime file.
Keep Kaggle CLI optional and external to the application image. Add ignored
paths for local `mlruns/`, downloaded Kaggle archives, and generated House
bundles while leaving `tests/fixtures/` trackable.

Create a small fixture with every declared feature, `Id`, positive numeric
`SalePrice`, at least one missing training value, and categories sufficient to
exercise unknown-category handling. Label it in comments/docs as test-only.

### Step 4: Run the focused test

```bash
python3 -m pytest tests/test_house_package.py -q
python3 -m ruff check src/house_pricing_mlops tests/test_house_package.py
```

Expected: PASS.

### Step 5: Commit

```bash
git add pyproject.toml requirements.txt requirements-dev.txt requirements-training.txt .gitignore .dockerignore src/house_pricing_mlops tests/fixtures/house_prices/train.csv tests/test_house_package.py
git commit -m "feat: add house pricing package boundary"
```

## Task 2: Implement schema and dataset provenance contracts

**Files:**

- Create: `src/house_pricing_mlops/schema.py`
- Create: `src/house_pricing_mlops/provenance.py`
- Create: `tests/test_house_schema.py`
- Create: `tests/test_house_provenance.py`

### Step 1: Write failing schema tests

Cover exact raw columns including `SalePrice`, missing target detection,
positive finite numeric targets, `Id` removal, exact numeric/categorical
partition, unexpected model columns, missing required values, non-empty
categorical strings, and valid ordered feature mappings.

### Step 2: Run to verify failure

```bash
python3 -m pytest tests/test_house_schema.py tests/test_house_provenance.py -q
```

Expected: FAIL with missing module/symbol errors.

### Step 3: Implement the minimal schema API

Define immutable schema constants and stable validation exceptions. Provide
functions equivalent to:

```python
validate_raw_frame(frame) -> ValidatedDataset
validate_prediction_frame(frame, allow_labels: bool) -> BatchValidationResult
validate_single_features(mapping) -> dict[str, object]
```

Return safe field-level errors without serializing row contents. Preserve row
indices only for in-memory diagnostics; never log identifiers or raw rows.

Implement provenance helpers for source URL, dataset version, UTC download
timestamp, SHA-256, row/column counts, feature schema, target, Git SHA, and
training configuration. Require explicit metadata for production input; use a
fixture-specific source marker only in tests.

### Step 4: Run focused tests

```bash
python3 -m pytest tests/test_house_schema.py tests/test_house_provenance.py -q
```

Expected: PASS.

### Step 5: Commit

```bash
git add src/house_pricing_mlops/schema.py src/house_pricing_mlops/provenance.py tests/test_house_schema.py tests/test_house_provenance.py
git commit -m "feat: define house pricing data contracts"
```

## Task 3: Build deterministic preprocessing, training, and metrics

**Files:**

- Create: `src/house_pricing_mlops/training.py`
- Create: `tests/test_house_training.py`
- Create: `tests/test_house_metrics.py`

### Step 1: Write failing training tests

Cover deterministic split indices, seed sensitivity, training-only
preprocessing statistics, numeric/categorical imputation, unknown categories,
the DummyRegressor/RandomForestRegressor/GradientBoostingRegressor candidate
set, complete pipeline output, and MAE/RMSE/R2/RMSLE metrics.

### Step 2: Run to verify failure

```bash
python3 -m pytest tests/test_house_training.py tests/test_house_metrics.py -q
```

Expected: FAIL because training functions do not exist.

### Step 3: Implement the minimal training API

Use a fixed configuration dataclass containing seed, split sizes, model
parameters, and target transform. Split first, then construct a
`ColumnTransformer` with numeric median imputation and categorical constant
missing-value imputation followed by
`OneHotEncoder(handle_unknown="ignore")`.

Wrap each estimator in a complete fitted target/prediction pipeline using
`log1p`/`expm1`. Fit only on the training partition. Select by validation RMSE
with deterministic candidate-order tie breaking. Evaluate the selected pipeline
once against the untouched test partition. Keep the fitted pipeline, metrics,
parameters, feature order, split metadata, and seed in the result.

### Step 4: Run focused tests

```bash
python3 -m pytest tests/test_house_training.py tests/test_house_metrics.py -q
```

Expected: PASS, including a test that changes validation-only values without
changing fitted training preprocessing statistics.

### Step 5: Commit

```bash
git add src/house_pricing_mlops/training.py tests/test_house_training.py tests/test_house_metrics.py
git commit -m "feat: add deterministic house pricing training"
```

## Task 4: Add MLflow tracking and the reproducible training entry point

**Files:**

- Modify: `src/house_pricing_mlops/training.py`
- Create: `scripts/train_house_price_model.py`
- Create: `tests/test_house_training_script.py`
- Create: `tests/test_house_mlflow.py`

### Step 1: Write failing tracking/script tests

Cover missing input failure without fabricated metrics, invalid dataset failure
before logging, explicit dataset version/seed, all required MLflow parameters
and metrics, local tracking fallback, configured remote URI handling, and safe
run summary output.

### Step 2: Run to verify failure

```bash
python3 -m pytest tests/test_house_training_script.py tests/test_house_mlflow.py -q
```

Expected: FAIL because the entry point and tracking boundary do not exist.

### Step 3: Implement tracking and CLI

Add a small MLflow adapter that creates one experiment, logs parameters and
metrics, records provenance tags, and logs the generated bundle after bundle
creation. Keep MLflow imports out of the Streamlit Iris path.

Implement CLI arguments for local input, exact S3 input coordinates, dataset
version/checksum metadata, output directory, experiment name, seed, model
version, and optional tracking URI. Require `train.csv`; never download from an
unspecified mutable URL inside the training script.

### Step 4: Run focused tests

```bash
python3 -m pytest tests/test_house_training_script.py tests/test_house_mlflow.py -q
python3 scripts/train_house_price_model.py --help
```

Expected: PASS and a usable help screen. Running without input must fail clearly.

### Step 5: Commit

```bash
git add src/house_pricing_mlops/training.py scripts/train_house_price_model.py tests/test_house_training_script.py tests/test_house_mlflow.py .gitignore
git commit -m "feat: track house pricing training with mlflow"
```

## Task 5: Create and validate reference profiles and drift scores

**Files:**

- Create: `src/house_pricing_mlops/drift.py`
- Create: `tests/test_house_drift.py`

### Step 1: Write failing drift tests

Cover numeric reference statistics/bins, categorical distributions, near-zero
unchanged scores, shifted numeric/categorical scores, unknown-category counts,
invalid-row counts, threshold-based drift counts, and the one-row sample-size
caveat.

### Step 2: Run to verify failure

```bash
python3 -m pytest tests/test_house_drift.py -q
```

Expected: FAIL because drift functions do not exist.

### Step 3: Implement the minimal profile and scoring API

Implement `build_reference_profile(training_features)` and
`score_batch(reference_profile, batch_features)`. Use a documented PSI-style
calculation over fixed numeric bins and categorical proportions, safe zero
handling, the educational warning threshold `0.20`, and explicit sample-size
caveats. Return per-feature score/flag, unknown-category count, and
invalid-row count.

### Step 4: Run focused tests

```bash
python3 -m pytest tests/test_house_drift.py -q
```

Expected: PASS.

### Step 5: Commit

```bash
git add src/house_pricing_mlops/drift.py tests/test_house_drift.py
git commit -m "feat: add house pricing drift monitoring"
```

## Task 6: Implement the complete model bundle and trusted loader

**Files:**

- Create: `src/house_pricing_mlops/bundle.py`
- Create: `tests/test_house_bundle.py`
- Create: `tests/test_house_model_loading.py`

### Step 1: Write failing bundle tests

Cover exact bundle members, a full fitted pipeline in `model.joblib`, manifest
fields/checksums, member mismatch, archive traversal/symlink/size/unexpected
member rejection, canonical bundle checksum validation, missing manifest fields,
and trusted in-memory loading.

### Step 2: Run to verify failure

```bash
python3 -m pytest tests/test_house_bundle.py tests/test_house_model_loading.py -q
```

Expected: FAIL because bundle creation/loading is absent.

### Step 3: Implement bundle creation and loading

Create deterministic JSON files with sorted keys and stable separators. Create a
deterministic tar archive with normalized metadata. Validate paths, member sizes,
checksums, and the canonical checksum before deserialization. The canonical
checksum is computed over a sorted member stream with the manifest checksum
field normalized before hashing; the loader reproduces that process.

### Step 4: Run focused tests

```bash
python3 -m pytest tests/test_house_bundle.py tests/test_house_model_loading.py -q
```

Expected: PASS.

### Step 5: Commit

```bash
git add src/house_pricing_mlops/bundle.py tests/test_house_bundle.py tests/test_house_model_loading.py
git commit -m "feat: add verified house pricing model bundles"
```

## Task 7: Add House configuration, prediction services, and logging

**Files:**

- Create: `src/house_pricing_mlops/config.py`
- Create: `src/house_pricing_mlops/prediction.py`
- Create: `src/house_pricing_mlops/logging_utils.py`
- Create: `tests/test_house_config.py`
- Create: `tests/test_house_prediction.py`
- Create: `tests/test_house_logging.py`

### Step 1: Write failing tests

Cover optional House-only settings, exact S3 coordinates, valid prediction
metadata/latency, invalid values before model calls, no raw input logging,
required `house_prediction` fields, all requested EMF metric names, allowed
dimensions only, and startup/model-load/validation/error metadata.

### Step 2: Run to verify failure

```bash
python3 -m pytest tests/test_house_config.py tests/test_house_prediction.py tests/test_house_logging.py -q
```

Expected: FAIL because House configuration, service, and logger are absent.

### Step 3: Implement minimal services

Keep configuration separate from `iris_mlops.config`. Add a House logger with
`Environment`, `Product`, and `ModelVersion` dimensions and no input
serialization. Implement single prediction around the trusted loaded bundle,
validate finite numeric ranges/non-empty categories, measure latency, and emit
the exact online event and metrics contract.

### Step 4: Run focused tests

```bash
python3 -m pytest tests/test_house_config.py tests/test_house_prediction.py tests/test_house_logging.py -q
```

Expected: PASS.

### Step 5: Commit

```bash
git add src/house_pricing_mlops/config.py src/house_pricing_mlops/prediction.py src/house_pricing_mlops/logging_utils.py tests/test_house_config.py tests/test_house_prediction.py tests/test_house_logging.py
git commit -m "feat: add house pricing serving contracts"
```

## Task 8: Add secure batch prediction and CSV output

**Files:**

- Modify: `src/house_pricing_mlops/prediction.py`
- Create: `tests/test_house_batch.py`

### Step 1: Write failing batch tests

Cover valid CSV output, optional `Id`, optional `SalePrice` metrics, missing and
unexpected columns, malformed CSV, wrong extension, file/row limits, invalid
row counts, required output columns, batch logging, and drift diagnostics.

### Step 2: Run to verify failure

```bash
python3 -m pytest tests/test_house_batch.py -q
```

Expected: FAIL because batch behavior is absent.

### Step 3: Implement the minimal batch boundary

Accept a file-like object, enforce a byte limit before parsing, parse with
pandas, validate the exact schema, process valid rows, calculate optional
metrics, score drift, and return a safe result plus CSV bytes. Use UTC ISO
timestamps and never include raw rows in errors or logs.

### Step 4: Run focused tests

```bash
python3 -m pytest tests/test_house_batch.py -q
```

Expected: PASS.

### Step 5: Commit

```bash
git add src/house_pricing_mlops/prediction.py tests/test_house_batch.py
git commit -m "feat: add safe house pricing batch predictions"
```

## Task 9: Integrate the House page without changing Iris behavior

**Files:**

- Modify: `app.py`
- Modify: `tests/test_app.py`
- Create: `tests/test_house_app.py`

### Step 1: Write failing Streamlit tests

Cover Iris as the default, existing Iris inputs/prediction/provenance,
12-feature House rendering, lazy House loading, no House configuration access
on Iris selection, House-specific load errors, versions/metrics/disclaimer,
online errors, batch upload/download, and labeled versus unlabeled batches.

### Step 2: Run to verify failure

```bash
python3 -m pytest tests/test_app.py tests/test_house_app.py -q
```

Expected: existing Iris tests pass where unchanged and new House tests fail.

### Step 3: Implement lazy navigation and pages

Add a sidebar selector defaulting to Iris. Keep the existing Iris page in its
branch. Put House imports/configuration/model loading behind the House branch or
lazy function. Cache House resources by exact settings/model coordinates and
add safe Streamlit adapters for batch results and downloads.

### Step 4: Run focused and full app tests

```bash
python3 -m pytest tests/test_app.py tests/test_house_app.py -q
python3 -m pytest tests/test_app.py tests/test_config.py tests/test_prediction.py tests/test_logging.py -q
```

Expected: PASS with Iris regression coverage intact.

### Step 5: Commit

```bash
git add app.py tests/test_app.py tests/test_house_app.py
git commit -m "feat: add lazy house pricing streamlit page"
```

## Task 10: Add dataset intake and real Kaggle provenance workflow

**Files:**

- Create: `scripts/intake_house_price_dataset.py`
- Create: `tests/test_house_intake.py`
- Create: `docs/house-pricing.md`
- Modify: `README.md`

### Step 1: Write failing intake tests

Cover exact `train.csv` acceptance, missing/duplicate target failure, invalid
target failure, checksum/timestamp/row-count/schema recording, no credential
output, fixture-specific source marking, intended S3 key metadata, and clear
failure when real input is absent.

### Step 2: Run to verify failure

```bash
python3 -m pytest tests/test_house_intake.py -q
```

Expected: FAIL because the intake script does not exist.

### Step 3: Implement intake

Accept an already downloaded Kaggle archive or local `train.csv`. Document the
external command:

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques -p /tmp/house-prices
```

Extract to a temporary directory, require the exact filename, validate the raw
schema, compute SHA-256, and write provenance JSON outside the repository. Do
not automatically upload until the operator supplies the exact bucket and
confirms the target version.

### Step 4: Run focused tests and documentation checks

```bash
python3 -m pytest tests/test_house_intake.py -q
python3 -m ruff check scripts/intake_house_price_dataset.py
rg -n "Kaggle|SalePrice|house-price-v1|datasets/house-prices/v1" README.md docs/house-pricing.md
```

Expected: PASS and clear documentation.

### Step 5: Commit

```bash
git add scripts/intake_house_price_dataset.py tests/test_house_intake.py docs/house-pricing.md README.md
git commit -m "docs: document house pricing dataset intake"
```

## Task 11: Add model upload/registry workflow and bundle integration

**Files:**

- Modify: `scripts/train_house_price_model.py`
- Create: `scripts/promote_house_price_model.py`
- Create: `tests/test_house_registry.py`
- Modify: `docs/house-pricing.md`

### Step 1: Write failing registry tests

Cover selected-run registration as `house-price-model`, explicit `candidate`
metadata without changing `champion`, approval-gated promotion with who/when/
why, exact S3 key/VersionId retention, no mutable latest alias, and rollback
metadata pointing to a prior immutable model.

### Step 2: Run to verify failure

```bash
python3 -m pytest tests/test_house_registry.py -q
```

Expected: FAIL because registry/promotion behavior is absent.

### Step 3: Implement the registry boundary

Keep registry calls behind an adapter so tests use a local fake and no AWS.
Record MLflow run ID, model version, dataset checksum, bundle checksum, S3 key,
and S3 VersionId. Require an explicit promotion command for `champion`.

### Step 4: Run focused tests

```bash
python3 -m pytest tests/test_house_registry.py -q
```

Expected: PASS.

### Step 5: Commit

```bash
git add scripts/train_house_price_model.py scripts/promote_house_price_model.py tests/test_house_registry.py docs/house-pricing.md
git commit -m "feat: add explicit house pricing model promotion"
```

## Task 12: Extend Docker and PR verification without Kaggle/AWS

**Files:**

- Modify: `Dockerfile`
- Modify: `scripts/docker_smoke_test.py` if needed
- Modify: `.github/workflows/pull-request.yml`
- Modify: `tests/test_docker_packaging.py`
- Modify: `tests/test_workflows.py`

### Step 1: Write failing static workflow tests

Preserve the PR trigger, read-only permissions, existing Iris tests/Ruff, no
AWS credentials, no live dataset requirement, and Docker build. Add assertions
for fixture training/schema/bundle/drift/Streamlit tests, dependencies, Docker
smoke test, and absence of Kaggle credentials.

### Step 2: Run to verify failure

```bash
python3 -m pytest tests/test_workflows.py tests/test_docker_packaging.py -q
```

Expected: FAIL because the workflow/image contracts are not extended.

### Step 3: Implement CI/image changes

Keep the runtime image free of tests, datasets, credentials, MLflow runs, and
model bundles. Install runtime requirements only in Docker. Extend PR CI to
install training requirements without credentials and run:

```bash
python3 -m pytest -q
python3 -m ruff check .
docker build --pull --file Dockerfile --tag house-mlops:pr-${GITHUB_SHA} .
```

Run Docker health smoke tests with House unconfigured and confirm Iris startup
and error handling remain safe.

### Step 4: Run verification

```bash
python3 -m pytest tests/test_workflows.py tests/test_docker_packaging.py -q
python3 -m ruff check .
docker build --pull --file Dockerfile --tag house-mlops:local .
```

Expected: PASS, subject to Docker daemon availability for the final command.

### Step 5: Commit

```bash
git add Dockerfile scripts/docker_smoke_test.py .github/workflows/pull-request.yml tests/test_docker_packaging.py tests/test_workflows.py
git commit -m "ci: verify house pricing without live credentials"
```

## Task 13: Add the manual House training workflow

**Files:**

- Create: `.github/workflows/train-house-price.yml`
- Create: `tests/test_house_training_workflow.py`
- Modify: `docs/house-pricing.md`

### Step 1: Write failing workflow tests

Assert manual or narrow dataset/code triggers, GitHub OIDC, exact S3 dataset
VersionId input, training dependency installation, deterministic training,
MLflow logging, bundle upload, model registration, metrics/version summary,
explicit production approval, and no static AWS keys or automatic deployment.

### Step 2: Run to verify failure

```bash
python3 -m pytest tests/test_house_training_workflow.py -q
```

Expected: FAIL because the workflow does not exist.

### Step 3: Implement the workflow

Use a dedicated repository/environment variable for the training role ARN and
restrict its trust to this repository and intended workflow/ref. Require:

```text
HOUSE_DATASET_S3_BUCKET
HOUSE_DATASET_S3_KEY
HOUSE_DATASET_S3_VERSION_ID
HOUSE_MODEL_S3_BUCKET
HOUSE_MODEL_S3_KEY
MLFLOW_TRACKING_URI
```

Write the S3 VersionId, bundle checksum, MLflow run ID, selected model, and
metrics to the GitHub summary. Make production promotion a protected
environment/manual approval step.

### Step 4: Run focused tests

```bash
python3 -m pytest tests/test_house_training_workflow.py -q
```

Expected: PASS.

### Step 5: Commit

```bash
git add .github/workflows/train-house-price.yml tests/test_house_training_workflow.py docs/house-pricing.md
git commit -m "ci: add manual house pricing training workflow"
```

## Task 14: Extend deployment workflow and IAM contracts

**Files:**

- Modify: `.github/workflows/deploy.yml`
- Create: `tests/test_house_deployment_workflow.py`
- Create: `docs/aws-house-pricing.md`
- Modify: `docs/house-pricing.md`

### Step 1: Write failing deployment/IAM tests

Assert existing Iris variables and exact VersionId remain present; House
bucket/key/VersionId/model/dataset variables are separate; image tag remains
`github.sha`; ECS service and health path remain unchanged; deployment summary
reports both model versions; no static keys are introduced; task-role House
read access is prefix/key scoped; and OIDC trust remains restricted.

### Step 2: Run to verify failure

```bash
python3 -m pytest tests/test_house_deployment_workflow.py -q
```

Expected: FAIL because House deployment variables/contracts are absent.

### Step 3: Implement workflow/IAM changes

Extend the primary container environment only with non-secret House coordinates
and metadata. Extend the ECS task-role inline policy with
`s3:GetObject`/`s3:GetObjectVersion` for the exact House model key while leaving
the Iris statement unchanged. Use a separate training-role contract for
dataset read/model write/MLflow operations.

Document account, Region, bucket, ECS service, log group, role names, exact
model key/version, rollback, and cleanup commands without secrets.

### Step 4: Run focused tests

```bash
python3 -m pytest tests/test_house_deployment_workflow.py -q
```

Expected: PASS.

### Step 5: Commit

```bash
git add .github/workflows/deploy.yml tests/test_house_deployment_workflow.py docs/aws-house-pricing.md docs/house-pricing.md
git commit -m "ci: deploy house pricing through existing ecs service"
```

## Task 15: Intake and version the real Kaggle dataset

**Files outside Git:**

- Temporary download directory such as `/tmp/house-prices`.
- S3 object `datasets/house-prices/v1/train.csv`.
- S3 provenance object under `datasets/house-prices/v1/`.

### Step 1: Verify external credentials without printing them

Run a safe command such as:

```bash
kaggle competitions files -c house-prices-advanced-regression-techniques
```

If Kaggle authentication is unavailable, stop and report the prerequisite.

### Step 2: Download to a temporary directory

```bash
mkdir -p /tmp/house-prices
kaggle competitions download -c house-prices-advanced-regression-techniques -p /tmp/house-prices
```

Extract outside the repository and do not print or commit credentials.

### Step 3: Run intake and inspect metadata

```bash
python3 scripts/intake_house_price_dataset.py \
  --input /tmp/house-prices/train.csv \
  --dataset-version v1 \
  --output-dir /tmp/house-prices/intake
```

Verify the recorded source URL, timestamp, checksum, 1460 rows, 81 raw
columns, `SalePrice` target, selected schema, and current Git SHA. Do not claim
production metrics yet.

### Step 4: Upload with the existing private versioned bucket

Use an explicitly authorized AWS session or approved training workflow to upload
the CSV and provenance JSON. Capture and record the returned S3 VersionId; never
rely on `latest`.

### Step 5: Verify the dataset boundary

Do not add the CSV or archive. Run `git status --ignored` and confirm the real
dataset remains outside Git.

## Task 16: Train, register, and publish the real model

**Files outside Git:**

- MLflow run and registered model `house-price-model`.
- S3 object `models/house-price/v1/model.tar.gz` and exact VersionId.
- S3 provenance/metrics metadata.

### Step 1: Train against the exact dataset version

Run the training command with the S3 bucket/key/VersionId, fixed seed, model
configuration, Git SHA, and output directory. Confirm it refuses missing or
changed checksums.

### Step 2: Verify run and artifact locally

Confirm all candidates were compared, selection is based on validation RMSE,
test metrics use the untouched test split, MLflow contains required tags and
metrics, and the bundle loader validates successfully.

### Step 3: Upload the immutable model bundle

Upload to `models/house-price/v1/model.tar.gz`, capture the S3 VersionId, and
verify object checksum/size with `head-object` and a versioned `get-object`.

### Step 4: Register candidate and explicitly promote

Register the run as `candidate`. Record a human-approved promotion decision
before assigning `champion`. Do not mutate an existing bundle in place.

### Step 5: Update deployment variables

Set exact House model bucket/key/VersionId and model/dataset versions in the
protected deployment environment. Keep Iris variables unchanged.

## Task 17: Deploy and perform production verification

**Files:**

- Modify only protected AWS/GitHub deployment configuration through approved
  workflows, not source-controlled secrets.
- Update: `docs/aws-house-pricing.md` with verified outputs.

### Step 1: Run all local verification

```bash
python3 -m pytest -q
python3 -m ruff check .
git diff --check
```

Expected: zero test failures, zero Ruff errors, and clean diff formatting.

### Step 2: Deploy the immutable image

Run the main deployment workflow with exact Iris and House coordinates. Wait for
ECS Express deployment success and record image tag/digest, service revision,
model versions, and endpoint.

### Step 3: Verify public behavior

Check the health endpoint, Iris prediction, House prediction, valid and invalid
batch CSV behavior, labeled metrics, unlabeled no-accuracy behavior, and the
one-row no-drift claim.

### Step 4: Verify observability

Inspect `/aws/ecs/iris-mlops` for startup, model-load, online, batch, and
validation JSON events. Confirm House EMF metrics use only the three allowed
dimensions and Iris alarms remain intact.

### Step 5: Record rollback and cleanup

Document the previous service revision/image, previous House S3 VersionId,
rollback command, model-promotion reversal, MLflow cleanup considerations, and
AWS cleanup commands. Do not delete resources as part of verification.

## Task 18: Final review and handoff

### Step 1: Inspect the final change set

```bash
git status --short --branch
git diff main...HEAD --stat
git diff main...HEAD -- .gitignore .dockerignore Dockerfile app.py .github src tests scripts docs
git ls-files | rg -i '(^|/)(train\.csv|.*\.joblib|.*\.pkl|.*\.tar\.gz|kaggle\.json|\.env)' || true
```

Confirm no dataset, credentials, or generated model artifact is tracked.

### Step 2: Run the complete verification suite

```bash
python3 -m pytest -q
python3 -m ruff check .
git diff --check
```

Run Docker and live AWS checks only when their prerequisites are available;
report blocked checks separately instead of treating them as passed.

### Step 3: Prepare the handoff

Report branch/commit range, Iris regression evidence, fixture and real-data
training evidence, dataset/model/MLflow/image versions, endpoint/health result,
CloudWatch evidence, changed AWS resources, limitations, rollback, and cleanup.

### Step 4: Commit final documentation updates

```bash
git add README.md docs/house-pricing.md docs/aws-house-pricing.md
git commit -m "docs: document house pricing mlops operations"
```
