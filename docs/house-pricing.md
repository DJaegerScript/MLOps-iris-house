# House Pricing MLOps guide

House Pricing is the second AI product in this repository. It demonstrates a
training-to-production lifecycle alongside the pre-trained Iris product:

```text
Iris: dataset/model artifact -> validate -> package -> deploy -> observe

House Pricing: dataset -> validate -> preprocess -> train -> evaluate
              -> track -> register -> deploy -> batch predict -> monitor drift
```

The House model uses the Kaggle [House Prices: Advanced Regression Techniques
data](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).
The repository contains only a small test fixture; the real dataset and model
bundle are stored outside Git in the private versioned S3 bucket. Kaggle
credentials and downloaded files are not committed.

## Dataset intake and provenance

Download the data outside the repository with an authenticated Kaggle CLI
session:

```bash
mkdir -p /tmp/house-prices
kaggle competitions download \
  -c house-prices-advanced-regression-techniques \
  -p /tmp/house-prices
```

The archive must contain the exact file `train.csv`. Validate it and write a
provenance-only record outside the repository:

```bash
python scripts/intake_house_price_dataset.py \
  --input /tmp/house-prices/house-prices-advanced-regression-techniques.zip \
  --dataset-version v1 \
  --output-dir /tmp/house-prices/intake \
  --s3-bucket "$HOUSE_DATASET_S3_BUCKET"
```

The command records:

- source URL: the Kaggle competition data page above;
- dataset version, UTC download timestamp, and SHA-256 checksum;
- row and raw-column counts;
- selected feature schema and `SalePrice` target;
- Git commit SHA and training configuration;
- intended immutable object key: `datasets/house-prices/v1/train.csv`.

The command does not upload data. After review, upload the local file to the
private, versioned project bucket and record the returned S3 `VersionId`. Never
use a mutable `latest` key for training or serving.

For tests, the fixture may be passed directly to the intake function. It uses
the explicit `fixture://house-prices/test-only` source marker. Fixture output
is not a production dataset or production result.

The verified real intake record is:

| Field | Value |
| --- | --- |
| Dataset source | Kaggle competition data page above |
| Dataset version | `v1` |
| Download timestamp | `2026-09-21T12:39:17.819367+00:00` |
| SHA-256 | `1e18addf81e5e4d347cc17ee6075bbe4a42b7fa26b9e5b063e8f692a5f929d41` |
| Rows / raw columns | `1460 / 81` |
| Target | `SalePrice` |
| S3 dataset key | `datasets/house-prices/v1/train.csv` |
| S3 dataset VersionId | `2IzOEQWUuttBf0c5gkW0X8kR5cr6j7d9` |
| S3 provenance VersionId | `lF2fjHc5TkjPN6m7aL9E2ZwqW9Gb3Twk` |
| Intake Git SHA | `51f6bfc1c839dbc58e7b54dfbe5b1e96cab9e42a` |

## Model schema and validation

The online model schema version is `house-price-v1`.

Numeric features:

`OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `1stFlrSF`,
`YearBuilt`, `FullBath`, `TotRmsAbvGrd`, and `GarageArea`.

Categorical features:

`Neighborhood`, `KitchenQual`, and `CentralAir`.

The online numeric guardrails are: `OverallQual` from 1 through 10,
`YearBuilt` from 1800 through 2100, and all other numeric values at least 0.
These are input safety checks for this educational product, not universal
real-estate validity rules. Categorical values are required to be non-empty;
the fitted encoder handles unseen categories and the batch drift report counts
them.

`Id` is accepted for dataset identity and optional batch output but is never a
model feature. `SalePrice` is required for training and optional for batch
prediction. Missing online values are rejected with a field-level validation
error; they are not silently replaced with arbitrary values. Training uses a
training-split-only numeric median and an explicit categorical missing token.

Batch CSV files must contain exactly the twelve model features and may also
contain `Id` and `SalePrice`. Unexpected columns, malformed CSVs, unsupported
file types, oversized uploads, excessive row counts, invalid numeric values,
and empty categorical values are rejected or reported without logging raw rows.

## Local training and experiment tracking

Install training dependencies in a development environment, not the runtime
image:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -r requirements-training.txt
```

Run deterministic fixture training as a local exercise:

```bash
python scripts/train_house_price_model.py \
  --input tests/fixtures/house_prices/train.csv \
  --dataset-version fixture-v1 \
  --output-dir /tmp/house-prices/training
```

The pipeline validates the raw schema, splits train/validation/test data with a
fixed seed, fits preprocessing only on the training split, compares a dummy
baseline, random forest, and gradient boosting candidate, selects by
validation RMSE, and evaluates the selected pipeline once on the untouched
test split. It records MAE, RMSE, R², and RMSLE when predictions are
non-negative.

The verified real training configuration used seed `42`, validation size
`0.15`, test size `0.15`, 40 random-forest estimators, 80 gradient-boosting
estimators, and the documented `log1p`/`expm1` target transformation. The
final code Git SHA was
`8291f15e7855d982df37438349ac72a35958fe83`.

MLflow uses a local file-backed tracking URI by default. A configured tracking
URI can be supplied for an approved managed tracking server. Tracked metadata
includes dataset version/checksum, Git SHA, feature list, preprocessing,
candidate/model parameters, metrics, duration, and seed. The registry and
production promotion must remain explicit and approval-gated; training must
not silently promote a model.

The training command writes a complete `model.tar.gz` bundle containing the
fitted preprocessing and prediction pipeline plus manifest, metrics, reference
profile, feature schema, and environment metadata. Register a selected run
only after the bundle has been uploaded to its immutable model key and the
returned S3 `VersionId` is known:

```bash
python scripts/promote_house_price_model.py \
  --registry /tmp/house-prices/registry.json \
  register-candidate \
  --summary /tmp/house-prices/training/training_summary.json \
  --s3-version-id "$HOUSE_MODEL_S3_VERSION_ID"
```

Registration creates a numbered version and a `candidate` alias. It never
creates a mutable `latest` alias and never changes `champion`. Promotion must
record an approver and reason:

```bash
python scripts/promote_house_price_model.py \
  --registry /tmp/house-prices/registry.json \
  promote --registry-version 1 \
  --approved-by "operator@example.com" \
  --reason "Reviewed test metrics and provenance"
```

Use the `rollback` command with the prior numbered version and an explicit
reason to restore a known immutable artifact. The registry keeps the prior
champion, approver, timestamp, and reason in its decision history.

The manual GitHub Actions workflow
`.github/workflows/train-house-price.yml` reads the exact dataset S3
`VersionId`, installs the training-only dependencies, logs to the configured
MLflow tracking URI, uploads the complete bundle, registers a candidate, and
writes the run/model/version/metric information to the workflow summary. A
separate protected `house-pricing-production` environment is required before
the optional promotion job can assign `champion`; training does not deploy the
application.

The verified real run selected `random_forest` by validation RMSE. Its
untouched-test metrics were MAE `17098.44`, RMSE `27672.43`, R² `0.8994`, and
RMSLE `0.1352`. The MLflow run ID was
`78c798da18f14485aea9f8376308b438`, registered model `house-price-model`
version `1`, and the local tracking snapshot was stored under the S3 `mlflow/`
prefix. The complete model bundle has checksum
`fad7b400105f1c41a5af2210761981c8c6cc84378a98d9299710a0997410f0d8` and is
stored at `models/house-price/v1/model.tar.gz` with S3 VersionId
`XWYuPZ9y7F2dzZNIubOZML49TgOjKl_D`.

## Application and monitoring

The Streamlit application keeps Iris as the default page and loads House
configuration and its exact versioned S3 bundle only after House Pricing is
selected. The House page provides single-property inputs, labeled or unlabeled
CSV batch prediction, model/dataset versions, metrics when labels exist, and
an Ames, Iowa educational-use warning.

Batch prediction produces `Id` when supplied, `predicted_sale_price`,
`model_version`, and `prediction_timestamp`. It also calculates a reference
profile drift score, reports drifted features and unknown categories, and
shows each per-feature score, invalid-row count, and unknown-category count. It
warns that one-row input is not a reliable drift sample. Numeric reference
statistics and categorical distributions are created during training.

Structured JSON and CloudWatch Embedded Metric Format logging are emitted by
the serving boundary. Low-cardinality dimensions are limited to environment,
product, and model version. Raw uploaded rows, addresses, identifiers, and
full prediction inputs are not logged.

## Current production boundary

The real Kaggle intake, S3 dataset version, MLflow run/model version, candidate
registration, and immutable model bundle publication are complete. The local
registry currently records candidate version `1`; no `champion` alias or
production promotion decision has been fabricated. ECS deployment remains
approval-gated. Before deploying, record the promotion decision, image tag and
digest, and ECS deployment revision in the AWS operations guide.

The application reuses the existing private S3 bucket, ECR repository, ECS
service, CloudWatch log group, and OIDC deployment pattern. House artifacts use
separate prefixes:

```text
datasets/house-prices/
models/house-price/
mlflow/
```

See the repository README and the implementation plan for the remaining
registry, CI/CD, IAM, deployment, rollback, and cleanup contracts.
