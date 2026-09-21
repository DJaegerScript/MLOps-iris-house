# AWS House Pricing operations

House Pricing is deployed as a second lazy page in the existing Iris
Streamlit service. The intended AWS boundary is account `163918295215`, Region
`ap-southeast-3`, and the existing private/versioned bucket
`iris-mlops-models-163918295215-apse3`.

Shared resources:

| Resource | Existing value |
| --- | --- |
| S3 bucket | `iris-mlops-models-163918295215-apse3` |
| ECR repository | `iris-mlops` |
| ECS cluster/service | `iris-mlops` / `iris-mlops` |
| CloudWatch log group | `/aws/ecs/iris-mlops` |
| Region | `ap-southeast-3` |

House-specific immutable prefixes are:

```text
datasets/house-prices/v1/train.csv
models/house-price/v1/model.tar.gz
mlflow/registry/house-price-model/<immutable-record>.json
```

## CodePipeline migration boundary

The CloudFormation template
`infra/cloudformation/house-mlops-pipelines.yml` provisions two CodePipeline V2
pipelines and four scoped CodeBuild projects:

- `house-pricing-model` is started with an exact dataset `VersionId`, trains
  with seed `42`, uploads an immutable bundle, registers a candidate, and
  automatically promotes it using the pipeline execution identity and reason.
  `ModelPromotionMode=manual` enables the optional SNS-backed manual approval
  action when a human gate is required.
- `house-pricing-app` follows the GitHub `main` branch, builds the commit SHA
  image, and deploys the existing ECS Express Mode service through the custom
  CodeBuild deployment action.

The stack creates only pipeline, artifact-bucket, notification-topic,
CodeBuild, and pipeline-role resources. The existing ECS service, execution
role, and task role are passed as parameters and remain externally managed.
The task role must retain the exact `house-pricing-model-read` statement for
the configured House model object; this stack does not broaden it or grant a
wildcard model prefix.

The GitHub CodeConnections ARN and MLflow Secrets Manager ARN are deployment
inputs. Values are not committed to the repository. The CodeBuild training
role reads the configured tracking secret, while model and application roles
receive only their pipeline-specific S3, ECR, SSM, ECS, and `iam:PassRole`
permissions.

The deployed application receives the exact House model S3 key and `VersionId`
through non-secret GitHub environment variables. The application does not
load a mutable `latest` object. The verified current coordinates are:

```text
dataset key:        datasets/house-prices/v1/train.csv
dataset VersionId:  2IzOEQWUuttBf0c5gkW0X8kR5cr6j7d9
model key:          models/house-price/v1/model.tar.gz
model VersionId:    XWYuPZ9y7F2dzZNIubOZML49TgOjKl_D
model version:      v1
dataset version:    v1
```

## IAM boundary

The existing ECS task role remains responsible for the Iris exact-object read.
The separate inline policy named `house-pricing-model-read` is installed and
verified on the same task role with only these actions and one exact resource:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:GetObjectVersion"],
      "Resource": "arn:aws:s3:::iris-mlops-models-163918295215-apse3/models/house-price/v1/model.tar.gz"
    }
  ]
}
```

Do not grant bucket-wide `*`, dataset write access, MLflow write access, or
access to unrelated prefixes to the ECS task role. The deployment workflow
checks this exact policy before updating ECS. Training uses a separate role for
exact dataset reads, model-bundle writes, and approved tracking/registry
operations.

## Deployment configuration

The existing `.github/workflows/deploy.yml` keeps the Iris variables and adds:

```text
HOUSE_MODEL_S3_BUCKET
HOUSE_MODEL_S3_KEY
HOUSE_MODEL_S3_VERSION_ID
HOUSE_MODEL_VERSION
HOUSE_DATASET_VERSION
HOUSE_ENVIRONMENT
```

AWS authentication remains GitHub OIDC through the existing deployment role;
no access key or secret is stored in GitHub variables. The workflow keeps the
commit SHA as the image tag, updates the existing ECS Express Mode service, and
waits for `SUCCESSFUL` deployment status and the existing `/_stcore/health`
path. Its summary reports both Iris and House versions.

## Verification and rollback

After a deployment, verify the public endpoint, Streamlit health, Iris sample
prediction, House sample prediction, labeled and unlabeled batch behavior,
invalid CSV handling, CloudWatch JSON events, and House EMF metrics. Keep the
prior image tag, prior House model `VersionId`, and prior champion registry
version together as the rollback record.

Rollback consists of restoring the previous immutable image tag and exact
House S3 `VersionId`, then waiting for the ECS deployment to become successful.
If the model itself is rolled back, use the explicit registry rollback command
and record the approver and reason. Never overwrite an object and never delete
the prior version before verification.

## CodePipeline operator walkthrough

The migration uses two separate pipelines. Start a model run with an exact
dataset object version; do not substitute `latest` or a mutable key reference:

```bash
aws codepipeline start-pipeline-execution \
  --name house-pricing-model \
  --region ap-southeast-3 \
  --variables \
    name=DATASET_VERSION,value=v1 \
    name=DATASET_S3_VERSION_ID,value="$DATASET_S3_VERSION_ID" \
    name=MODEL_VERSION,value=v2
```

The model pipeline trains with seed `42`, emits a candidate release artifact,
and automatically promotes it under the configured default. If the stack is
deployed with `ModelPromotionMode=manual`, review the candidate metrics,
bundle checksum, registry version, and model S3 `VersionId` in the SNS/manual
approval action before approving it. Promotion writes the champion release
pointer and starts `house-pricing-app` with the exact model coordinates.

For a code-only release, the app pipeline follows `main`, resolves the current
approved release once, builds the commit-tagged image, and deploys the existing
ECS Express service. The deployment verifies the task-role policy, waits for a
new `SUCCESSFUL` deployment, checks `/_stcore/health`, and produces a safe
deployment summary. The smoke harness reports only immutable coordinates,
statuses, counts, and latency:

```bash
python scripts/codepipeline/smoke_test_house_release.py \
  --public-url https://ir-c1c8733af56f4191b9c1d512f141541f.ecs.ap-southeast-3.on.aws \
  --approved-release /path/to/approved-release.json \
  --image-release /path/to/image-release.json \
  --region ap-southeast-3
```

The harness checks the public health endpoint, a safe Iris/UI check marker,
House online prediction, labeled and unlabeled CSV behavior, and invalid CSV
rejection. It uses an in-memory safe fixture and never reports rows, addresses,
identifiers, credentials, or prediction values. Use `--skip-model` only for a
health-only check when the operator does not have read access to the exact
House model object.

Keep the previous image digest and previous champion release manifest together
as the rollback record. Roll back by starting a normal app-pipeline execution
with explicit immutable overrides:

```bash
python scripts/codepipeline/rollback_house_release.py \
  --pipeline-name house-pricing-app \
  --previous-image-uri 163918295215.dkr.ecr.ap-southeast-3.amazonaws.com/iris-mlops:<previous-commit> \
  --previous-image-digest sha256:<64-lowercase-hex-characters> \
  --previous-release /path/to/previous-approved-release.json \
  --reason "restore the last verified House release" \
  --region ap-southeast-3
```

Rollback refuses mutable image references, non-champion releases, missing
reasons, and non-SHA-256 image digests. It invokes the app pipeline with the
prior image digest and exact House model S3 `VersionId`; it never overwrites or
deletes an S3 object and never changes the approved-release pointer.

After each release, inspect CodeBuild, ECR, ECS deployment, and CloudWatch
stages. Confirm Iris remains functional, House online and batch checks pass,
and CloudWatch contains the `HousePricingMLOps` EMF metrics. Record the image
digest, model `VersionId`, dataset version, deployment ARN, health latency, and
rollback reason in the release handoff without copying raw input data.

## Cost and cleanup

This feature reuses the existing ECS, ECR, S3, and CloudWatch resources. MLflow
tracking-server, SageMaker, ECS, CloudWatch, S3, ECR, and data-transfer charges
may apply if additional managed resources are enabled. Review the existing
`iris-mlops-monthly` budget before creating any new service.

Do not delete shared resources during feature verification. For eventual
cleanup, first confirm ownership, stop the ECS service, remove only the House
model/dataset/MLflow object versions, and retain the Iris object and shared
logging/alerting resources unless the whole project is intentionally retired.

The candidate-to-champion promotion was explicitly approved by `DJaegerScript`
for registry version `1`, with the decision recorded in the immutable S3 audit
object
`mlflow/registry/house-price-model/v1-champion-78c798da18f14485aea9f8376308b438.json`.
The production deployment completed successfully in GitHub Actions run
`35607090494` using image tag
`98105435797360ee5bb5e1e16de3b22cbe028088` and ECR digest
`sha256:4cbcfc177ab1a753e9cb8bc7302c1622db80f07fd5a8b1dad70780a242055539`.
The ECS deployment revision was
`arn:aws:ecs:ap-southeast-3:163918295215:service-deployment/iris-mlops/iris-mlops/Z27gI_k-WPAQ1Tpe4sEhv`.

Post-deployment verification confirmed the public health endpoint, Iris
setosa prediction, House online prediction, labeled and invalid House batch
validation, structured CloudWatch logs, and House Pricing custom metrics. The
training workflow remains manual and does not retrain during application
deployment.
