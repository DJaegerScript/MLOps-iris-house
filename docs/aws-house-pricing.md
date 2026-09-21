# AWS House Pricing operations

House Pricing is deployed as a separate House-only ECS Express service. Iris
remains deployed on the existing `iris-mlops` service and endpoint. The
intended AWS boundary is account `163918295215`, Region `ap-southeast-3`, and
the existing private/versioned bucket `iris-mlops-models-163918295215-apse3`.

Shared resources:

| Resource | Existing value |
| --- | --- |
| S3 bucket | `iris-mlops-models-163918295215-apse3` |
| ECR repository | `iris-mlops` |
| ECS cluster | `iris-mlops` |
| Iris ECS service | `iris-mlops` |
| House ECS service | `house-pricing` |
| House endpoint | `https://ho-f7043fa5170848dab5e1df01f8d590ad.ecs.ap-southeast-3.on.aws` |
| CloudWatch log group | `/aws/ecs/iris-mlops` |
| House log group | `/aws/ecs/house-pricing` |
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
  image, and creates or updates only the `house-pricing` ECS Express service
  through the custom CodeBuild deployment action.

The stack creates pipeline, artifact-bucket, notification-topic, CodeBuild,
House task-role, and House log-group resources. It reuses the existing ECS
cluster, execution role, and infrastructure role. The existing Iris ECS
service and Iris task role remain outside this House deployment path. The
House task role contains the exact `house-pricing-model-read` statement for
the configured House model object; it does not broaden access or grant a
wildcard model prefix.

The legacy GitHub OAuth source uses a GitHub token stored in Secrets Manager
and an AWS CodePipeline webhook with a separate generated HMAC secret. The
token must have `repo` and `admin:repo_hook` scopes; it is never committed or
printed. The verified account currently has no GitHub token secret and no
MLflow Secrets Manager secret, so the parameter file leaves those values
unresolved and the training buildspec uses an ephemeral local MLflow store
plus an immutable S3 tracking snapshot. If a managed MLflow endpoint is later
introduced, provide its secret ARN and enable the conditional training-role
permission.

The deployed application receives the exact House model S3 key and `VersionId`
through non-secret CodePipeline variables. The application does not load a
mutable `latest` object. The currently approved champion coordinates are:

```text
dataset key:        datasets/house-prices/v1/train.csv
dataset VersionId:  2IzOEQWUuttBf0c5gkW0X8kR5cr6j7d9
model key:          models/house-price/v1/model.tar.gz
model VersionId:    LjgOhvO7DCyHV7KoNMVGyJ0ux2Gui51X
model version:      v2
dataset version:    v1
```

## IAM boundary

The dedicated House ECS task role owns the House exact-object read. The inline
policy named `house-pricing-model-read` has only these actions and one exact
resource:

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

The existing `.github/workflows/deploy.yml` remains the Iris-owned deployment
path and continues to target the existing Iris service. House deployment is
handled by `house-pricing-app`, not by that workflow. The House task receives
`APP_VARIANT=house`, the exact approved model coordinates, and the
`/aws/ecs/house-pricing` log group. AWS authentication for the House pipeline
uses its scoped CodeBuild roles; no access key or secret is stored in GitHub.

## Verification and rollback

After a House deployment, verify the House endpoint, Streamlit health, House
sample prediction, labeled and unlabeled batch behavior, invalid CSV handling,
CloudWatch JSON events, and House EMF metrics. Separately confirm Iris remains
healthy on its existing endpoint. Keep the prior image tag, prior House model
`VersionId`, and prior champion registry version together as the rollback
record.

House rollback consists of restoring the previous immutable image tag and
exact House S3 `VersionId` through `house-pricing-app`, then waiting for the
House ECS deployment to become successful. It never targets `iris-mlops`.
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
approved release once, builds the commit-tagged image, and deploys only the
House ECS Express service. The deployment verifies the dedicated task-role
policy, waits for a new `SUCCESSFUL` deployment, checks `/_stcore/health`, and
produces a safe deployment summary. The smoke harness reports only immutable
coordinates, statuses, counts, and latency:

```bash
python scripts/codepipeline/smoke_test_house_release.py \
  --public-url https://ho-f7043fa5170848dab5e1df01f8d590ad.ecs.ap-southeast-3.on.aws \
  --approved-release /path/to/approved-release.json \
  --image-release /path/to/image-release.json \
  --region ap-southeast-3
```

The harness checks the House public health endpoint, a safe Iris/UI check
marker, House online prediction, labeled and unlabeled CSV behavior, and
invalid CSV rejection. It identifies the deployment service as
`house-pricing`, uses an in-memory safe fixture, and never reports rows,
addresses, identifiers, credentials, or prediction values. Use `--skip-model`
only for a health-only check when the operator does not have read access to the
exact House model object.

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

This feature reuses the existing ECS cluster, ECR, S3, and region while adding
one House ECS Express service and House log group. MLflow tracking-server,
SageMaker, ECS, CloudWatch, S3, ECR, and data-transfer charges may apply if
additional managed resources are enabled. Review the existing
`iris-mlops-monthly` budget before creating the House service.

Do not delete shared resources during feature verification. For eventual
cleanup, first confirm ownership, stop the ECS service, remove only the House
model/dataset/MLflow object versions, and retain the Iris object and shared
logging/alerting resources unless the whole project is intentionally retired.

The Iris service baseline is intentionally preserved by this House pipeline.
The separated House deployment completed through app-pipeline execution
`021813df-fa84-48d3-ad56-e41ec711d3fe`:

```text
House service ARN:       arn:aws:ecs:ap-southeast-3:163918295215:service/iris-mlops/house-pricing
House deployment ARN:    arn:aws:ecs:ap-southeast-3:163918295215:service-deployment/iris-mlops/house-pricing/EXsdro8Rs9T3dIOLon0qU
Image digest:             sha256:fcab7377c394878111c37ad4f62b429973470acb19369a262f78c2d64f70a98b
House model VersionId:    LjgOhvO7DCyHV7KoNMVGyJ0ux2Gui51X
Health:                   HTTP 200 / ok
Smoke:                    PASSED
```

The existing Iris service remains on its original ARN, endpoint, image,
environment, task role, log group, and `SUCCESSFUL` deployment revision. The
training workflow remains manual and does not retrain during application
deployment.

## Migration status

The separated implementation is deployed and verified. CloudFormation stack
`house-mlops-pipelines` is `UPDATE_COMPLETE`; the House task role, log group,
service, and endpoint are active. The House pipeline uses the legacy GitHub
OAuth source and webhook required for `ap-southeast-3`. No House pipeline
action targets the existing `iris-mlops/iris-mlops` service.
