# Iris MLOps showcase

This repository is a minimal MLOps product for serving a vetted, pre-trained
Iris classifier. It demonstrates the path from an external model artifact to a
public Streamlit application:

```text
external artifact intake -> checksum/schema validation -> S3 versioning
-> Docker packaging -> ECR -> ECS Express Mode/Fargate -> prediction logs
-> CloudWatch logs, EMF metrics, and alarms
```

House Pricing is intentionally out of scope. The application, CI workflow, and
deployment workflow never train a model.

## What this demonstrates

- Model artifact management and provenance
- Manifest and feature-schema validation
- Packaging and containerization
- Deployment to ECS Express Mode on Fargate
- Online serving through Streamlit
- Structured prediction logging and CloudWatch observability
- Git commit, image, and model version tracking

It intentionally does not demonstrate model training, experiment comparison,
retraining, automated retraining, or automated model promotion. There are no
SageMaker Training Jobs, Kubernetes resources, FastAPI service, RDS database,
API Gateway, or MLflow registry.

## Architecture

- `app.py` renders the four-input Streamlit product and caches one runtime model
  with `st.cache_resource`.
- `src/iris_mlops/manifest.py` validates the metadata-only v1 manifest.
- `src/iris_mlops/model.py` fetches one explicit S3 object version, verifies the
  bundle and each member checksum, then loads the scaler, LDA model, and label
  encoder in memory.
- `scripts/package_model_bundle.py` is an offline operator intake step. It
  packages only the four checksum-approved source artifacts and never downloads,
  deserializes, or uploads a model.
- S3 is private and versioned. The model bundle is not in GitHub or the Docker
  image. The dataset is not in GitHub, S3, or the image for this showcase.
- ECR stores the application image. ECS Express Mode provides the public HTTPS
  endpoint, Fargate task, load balancer, health check, and canary deployment.
- ECS sends stdout to `/aws/ecs/iris-mlops`; the application writes structured
  JSON and CloudWatch Embedded Metric Format (EMF).
- GitHub Actions uses short-lived AWS credentials through GitHub OIDC.

## Model provenance

The v1 artifact is the educational model published at
[`rajuamburu/iris-classifier`](https://huggingface.co/rajuamburu/iris-classifier),
fixed to revision `44b3ab8b94defd12e964f6327fa56c1efd7c4df6`. The source README
reports an LDA classifier, 10-fold cross-validation accuracy of 97.5%, test
accuracy of 100%, and 150 Iris samples across three classes. Those metrics are
recorded as source-reported values only; this project does not retrain or
reproduce them.

The source files are inspected and pinned by the manifest. Their SHA-256 values
are recorded in
[`artifacts/iris-classifier-v1.manifest.json`](artifacts/iris-classifier-v1.manifest.json):

- `iris_model.pkl`: `7774a99e5a30e7890fa1d75e5e2a44d14718f16b591334fde385d15ad03aa0a7`
- `scaler.pkl`: `10440ff9b1a4b0789ba496b6daffdb803e1bc68b1066a3dd82fcf2a45824daad`
- `label_encoder.pkl`: `59c14f9f5ffcd26188226746d0602c8552e0452045e6981138a8e946839ae45c`
- `metadata.pkl`: `7557328ea1eaa3c5e5a6bfd509e538a48f7f9e4a8905a0130f2ba6aa677ad05b`
- deterministic bundle: `a0da1d85be416055f09551cd3ef1070e21503a24f675c2dfd7077b27262a35dd`

The runtime is pinned to scikit-learn `1.8.0`; the source does not report the
Python version, so the manifest records `not reported by source`. The feature
schema is `sepal_length`, `sepal_width`, `petal_length`, `petal_width`. The
served class names are `setosa`, `versicolor`, and `virginica`; the source
labels are normalized from `Iris-*` by the documented manifest rule.

## Offline artifact intake and S3 storage

Keep the downloaded model files in a private temporary directory outside the
repository. After inspecting the immutable source revision, place exactly the
four source files there and run:

```bash
python scripts/package_model_bundle.py \
  --input-dir /private/path/containing-the-four-source-files \
  --output /tmp/iris-classifier-v1.tar.gz
```

The command checks every member against the manifest and checks the deterministic
bundle checksum. Upload the resulting bundle to the private, versioned S3 key
using an approved operator AWS session or the connected AWS integration. The
application requires the bucket, key, and exact S3 `VersionId`; it never loads a
mutable `latest` object.

The deployed artifact is currently:

```text
bucket:    iris-mlops-models-163918295215-apse3
key:       models/iris-classifier/v1/model.tar.gz
version:   OxDlS6cZLaVambx_QDpvdogy2RrRR4o3
sha256:    a0da1d85be416055f09551cd3ef1070e21503a24f675c2dfd7077b27262a35dd
```

No model binary, dataset, AWS access key, secret key, password, or token should
be committed. `.gitignore`, `.dockerignore`, and the Dockerfile enforce the
model/data boundary; only the metadata manifest is copied into the image.

## Local setup

Use Python 3.11 or newer and install the pinned runtime plus development tools:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

The application needs ambient AWS credentials with read access to the exact
versioned S3 object. Do not put credentials in `.env` committed to Git. Copy
`.env.example` to a private local `.env` only if your local process provides
these non-secret runtime coordinates:

```text
IRIS_MODEL_S3_BUCKET=iris-mlops-models-163918295215-apse3
IRIS_MODEL_S3_KEY=models/iris-classifier/v1/model.tar.gz
IRIS_MODEL_S3_VERSION_ID=OxDlS6cZLaVambx_QDpvdogy2RrRR4o3
IRIS_MODEL_VERSION=v1
IRIS_ENVIRONMENT=local
GIT_COMMIT_SHA=local
DOCKER_IMAGE_VERSION=local
LOW_CONFIDENCE_THRESHOLD=0.80
```

## Run Streamlit locally

```bash
streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port=8501
```

Open `http://127.0.0.1:8501`. The health endpoint is
`http://127.0.0.1:8501/_stcore/health` and should return `ok`. The application
loads the model once per Streamlit resource cache and never trains during
startup or inference.

## Docker

The production-oriented image is non-root, pinned to Python 3.12 and the
runtime dependency versions, exposes port 8501, has a health check, and does
not contain credentials or model binaries.

```bash
docker build --tag iris-mlops:local .
docker run --rm --publish 8501:8501 \
  --env-file .env \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  iris-mlops:local
```

Run the health-only smoke test with a Docker daemon available:

```bash
python scripts/docker_smoke_test.py \
  --image iris-mlops:local \
  --host-port 8510
```

The optional `--sample-prediction` mode requires the real S3 coordinates,
ambient AWS access, and Playwright. It submits the sample through the actual
Streamlit UI; it does not use a fixture model.

## Tests and checks

```bash
pytest -q
ruff check .
git diff --check
```

Tests cover manifest and schema validation, checksum-verified model loading,
valid and invalid predictions, class names, probability shape, structured
logging, Streamlit startup/error handling, Docker packaging, the offline intake
step, and the GitHub workflows. No test trains a model.

## AWS deployment

The deployed environment uses one Region: `ap-southeast-3`, account
`163918295215`, and project prefix `iris-mlops`. The AWS resources created for
this showcase are:

| Resource | Name / ARN |
| --- | --- |
| S3 model bucket | `iris-mlops-models-163918295215-apse3` / `arn:aws:s3:::iris-mlops-models-163918295215-apse3` |
| ECR repository | `iris-mlops` / `arn:aws:ecr:ap-southeast-3:163918295215:repository/iris-mlops` |
| ECS cluster | `arn:aws:ecs:ap-southeast-3:163918295215:cluster/iris-mlops` |
| ECS Express service | `arn:aws:ecs:ap-southeast-3:163918295215:service/iris-mlops/iris-mlops` |
| CloudWatch log group | `/aws/ecs/iris-mlops` |
| Model access role | `arn:aws:iam::163918295215:role/iris-mlops-ecs-task-role` |
| Execution role | `arn:aws:iam::163918295215:role/iris-mlops-ecs-execution-role` |
| Express infrastructure role | `arn:aws:iam::163918295215:role/iris-mlops-ecs-infrastructure-role` |
| GitHub OIDC role | `arn:aws:iam::163918295215:role/iris-mlops-github-actions-role` |

The public application URL is:

`https://ir-c1c8733af56f4191b9c1d512f141541f.ecs.ap-southeast-3.on.aws`

The live health endpoint returned `200 ok`, and the verified sample prediction
returned `setosa` with a 100% class probability. The deployed image is tagged
with commit `62a5c5668c0d896ab767915dbcf012463a915616` and resolves to ECR digest
`sha256:b7f91060c588be16127b73c8701561bcaf1b037e0c28fc0df88f86a7bc2e500f`.
The deployed model version is `v1`.

### GitHub Actions OIDC variables

The deployment workflow is `.github/workflows/deploy.yml`. Set these as GitHub
repository or production-environment variables, not static AWS credentials:

```text
AWS_REGION=ap-southeast-3
IRIS_AWS_DEPLOY_ROLE_ARN=arn:aws:iam::163918295215:role/iris-mlops-github-actions-role
IRIS_ECR_REPOSITORY=iris-mlops
IRIS_ECS_SERVICE_ARN=arn:aws:ecs:ap-southeast-3:163918295215:service/iris-mlops/iris-mlops
IRIS_ECS_SERVICE_NAME=iris-mlops
IRIS_ECS_EXECUTION_ROLE_ARN=arn:aws:iam::163918295215:role/iris-mlops-ecs-execution-role
IRIS_ECS_INFRASTRUCTURE_ROLE_ARN=arn:aws:iam::163918295215:role/iris-mlops-ecs-infrastructure-role
IRIS_ECS_TASK_ROLE_ARN=arn:aws:iam::163918295215:role/iris-mlops-ecs-task-role
IRIS_CLOUDWATCH_LOG_GROUP=/aws/ecs/iris-mlops
IRIS_MODEL_S3_BUCKET=iris-mlops-models-163918295215-apse3
IRIS_MODEL_S3_KEY=models/iris-classifier/v1/model.tar.gz
IRIS_MODEL_S3_VERSION_ID=OxDlS6cZLaVambx_QDpvdogy2RrRR4o3
IRIS_MODEL_VERSION=v1
IRIS_ENVIRONMENT=production
LOW_CONFIDENCE_THRESHOLD=0.80
```

The IAM trust policy accepts only the GitHub OIDC audience
`sts.amazonaws.com` and this subject:

```text
repo:DJaegerScript@85334514/MLOps-iris-house@1378975661:ref:refs/heads/main
```

Pull requests run dependency installation, tests, linting, and a Docker build.
Deployment runs only for `main` or a manual dispatch and contains no model
training step.

## CloudWatch logs, metrics, and alarms

ECS forwards stdout to `/aws/ecs/iris-mlops`. Each prediction includes JSON
fields for the event, model version, predicted class, confidence, latency, and
input validity. Startup, model-load success/failure, invalid inputs, and
prediction errors are also logged with the Git SHA and image version.

The EMF namespace is `IrisMLOps`. Metrics use only `Environment` and
`ModelVersion` dimensions:

- `PredictionCount`
- `PredictionErrorCount`
- `PredictionLatencyMs`
- `LowConfidencePredictionCount`
- `InvalidInputCount`
- `ModelLoadFailure`

Project alarms are:

- `iris-mlops-prediction-errors`
- `iris-mlops-model-load-failure`
- `iris-mlops-service-health` for the Express Mode ALB healthy target count

The SNS topic used by the alarms and budget is
`arn:aws:sns:ap-southeast-3:163918295215:iris-mlops-budget-alerts`. The AWS
account did not have a contact email available, so the topic currently has no
email subscriber; add an approved subscription before relying on email
notifications. The monthly budget is `iris-mlops-monthly` with a `$10` limit
and an 80% forecasted-spend notification.

## Rollback

Express Mode creates a new revision for each image update and performs a
canary deployment with automatic rollback on failed health/error monitoring.
For a manual rollback, update the service to the previous immutable image tag
and keep the same S3 coordinates:

```bash
aws ecs update-express-gateway-service \
  --region ap-southeast-3 \
  --service-arn arn:aws:ecs:ap-southeast-3:163918295215:service/iris-mlops/iris-mlops \
  --primary-container '{"image":"163918295215.dkr.ecr.ap-southeast-3.amazonaws.com/iris-mlops:<known-good-commit-sha>","containerPort":8501,"awsLogsConfiguration":{"logGroup":"/aws/ecs/iris-mlops","logStreamPrefix":"iris-mlops"}}'
```

Use the ECS deployment status and CloudWatch logs to confirm the rollback.
Do not overwrite an S3 object; select the intended immutable `VersionId`.

## Cleanup

These commands target only the resources listed above. Review the account and
Region first, and do not delete an IAM OIDC provider or SNS topic if it is
shared by another workload.

```bash
export AWS_REGION=ap-southeast-3
export AWS_ACCOUNT_ID=163918295215

aws ecs delete-express-gateway-service \
  --region "$AWS_REGION" \
  --service-arn arn:aws:ecs:ap-southeast-3:163918295215:service/iris-mlops/iris-mlops
aws ecr delete-repository --region "$AWS_REGION" --repository-name iris-mlops --force
aws logs delete-log-group --region "$AWS_REGION" --log-group-name /aws/ecs/iris-mlops
aws budgets delete-budget --account-id "$AWS_ACCOUNT_ID" --budget-name iris-mlops-monthly
aws sns delete-topic --region "$AWS_REGION" --topic-arn arn:aws:sns:ap-southeast-3:163918295215:iris-mlops-budget-alerts

# Empty all versions before deleting the private versioned bucket.
aws s3api list-object-versions \
  --region "$AWS_REGION" \
  --bucket iris-mlops-models-163918295215-apse3 \
  --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}' \
  --output json > /tmp/iris-mlops-object-versions.json
aws s3api delete-objects \
  --region "$AWS_REGION" \
  --bucket iris-mlops-models-163918295215-apse3 \
  --delete file:///tmp/iris-mlops-object-versions.json
aws s3api delete-bucket --region "$AWS_REGION" --bucket iris-mlops-models-163918295215-apse3

aws iam delete-role-policy --role-name iris-mlops-ecs-task-role --policy-name iris-mlops-model-read
aws iam delete-role-policy --role-name iris-mlops-github-actions-role --policy-name iris-mlops-github-actions
aws iam delete-role --role-name iris-mlops-ecs-task-role
aws iam delete-role --role-name iris-mlops-ecs-execution-role
aws iam delete-role --role-name iris-mlops-ecs-infrastructure-role
aws iam delete-role --role-name iris-mlops-github-actions-role
```

The ECS cluster and Express Mode managed load-balancer resources should be
checked after service deletion. Delete only project-created managed resources
that remain; leave shared default VPC, subnets, security groups, OIDC
providers, and service-linked roles untouched unless confirmed project-only.

## Scope and cost

This showcase uses one small Fargate task (0.5 vCPU / 1 GiB) with a maximum of
two tasks, one small ECR image, one private versioned S3 object, a CloudWatch
log group with 14-day retention, EMF custom metrics, three project alarms, and a
small monthly budget notification. Charges depend on uptime, Fargate runtime,
load-balancer usage, image storage, S3 requests/storage, CloudWatch ingestion,
and public data transfer. Stop or delete the service when it is not being used;
the budget alert is a guardrail, not a cost guarantee.
