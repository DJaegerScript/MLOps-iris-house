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

## Cost and cleanup

This feature reuses the existing ECS, ECR, S3, and CloudWatch resources. MLflow
tracking-server, SageMaker, ECS, CloudWatch, S3, ECR, and data-transfer charges
may apply if additional managed resources are enabled. Review the existing
`iris-mlops-monthly` budget before creating any new service.

Do not delete shared resources during feature verification. For eventual
cleanup, first confirm ownership, stop the ECS service, remove only the House
model/dataset/MLflow object versions, and retain the Iris object and shared
logging/alerting resources unless the whole project is intentionally retired.

The ECS deployment is still pending the explicit candidate-to-champion
promotion decision. The local AppTest and direct S3 serving checks already
verified Iris routing, House lazy loading, online prediction, labeled batch
metrics, and structured logs against the exact model VersionId above.
