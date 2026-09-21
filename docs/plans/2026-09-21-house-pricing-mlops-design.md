# House Pricing MLOps Design

Date: 2026-09-21

## Capability

Add House Pricing as a second AI product in the existing Streamlit application.
Iris remains a deployment-focused product backed by its existing version-pinned
artifact. House Pricing adds a reproducible training-to-production lifecycle:

```text
Kaggle train.csv -> provenance and S3 versioning -> schema validation
-> preprocessing -> candidate training -> MLflow tracking -> evaluation
-> model bundle and registry -> exact S3 artifact -> Streamlit prediction
-> batch prediction -> drift monitoring -> CloudWatch
```

The real dataset source is the Kaggle House Prices: Advanced Regression
Techniques competition:

<https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data>

The download is performed outside Git using Kaggle authentication. Credentials,
the full dataset, and trained binaries are never committed.

## Existing constraints

- The current branch starts at the verified clean `main` commit.
- The existing Iris code, artifact, S3 key/version, deployment, alarms, and
  CloudWatch namespace must remain compatible.
- Production AWS resources are in account `163918295215`, Region
  `ap-southeast-3`.
- The existing private, versioned S3 bucket is
  `iris-mlops-models-163918295215-apse3`.
- The existing ECS Express service remains the only public application service.
- House Pricing must be independently unavailable without preventing Iris from
  loading or predicting.
- The current AWS CLI identity is suitable for read-only inspection only. AWS
  mutations use the existing least-privilege GitHub OIDC deployment path or
  explicitly scoped operator permissions.

## Product and model contract

The House Pricing model schema version is `house-price-v1`.

Numeric features, in declared order:

```text
OverallQual, GrLivArea, GarageCars, TotalBsmtSF, 1stFlrSF,
YearBuilt, FullBath, TotRmsAbvGrd, GarageArea
```

Categorical features:

```text
Neighborhood, KitchenQual, CentralAir
```

The target is `SalePrice`. `Id` is provenance/row identity only and is removed
from model features. Batch uploads may also contain `Id` and `SalePrice`, but no
other columns are accepted.

Missing-value policy:

- Raw training data uses deterministic fitted imputers: numeric median and an
  explicit categorical missing value.
- The online form requires every declared feature and rejects missing or invalid
  values rather than applying arbitrary UI defaults.
- Batch rows with missing required values are reported as invalid and are not
  predicted. Valid rows are still processed when possible.
- Unknown categorical values are handled by
  `OneHotEncoder(handle_unknown="ignore")`, counted, and surfaced in batch
  diagnostics.

The default target strategy is `log1p` during fitting and `expm1` during
prediction through a fitted scikit-learn target wrapper. Metrics are reported
in original dollar units. RMSLE is reported only for non-negative predictions.

## Repository architecture

### House Pricing package

Create `src/house_pricing_mlops/` with narrow modules:

- `schema.py`: feature constants, raw-schema validation, target validation,
  single-row validation, and batch validation.
- `provenance.py`: dataset metadata, checksums, Git/configuration capture, and
  safe JSON serialization.
- `training.py`: deterministic splits, fitted preprocessing, candidate models,
  model selection, metrics, and MLflow logging boundary.
- `bundle.py`: bundle creation, canonical checksums, manifest validation, and
  trusted loading of the complete pipeline.
- `prediction.py`: online and batch prediction services and output contracts.
- `drift.py`: reference profile generation and educational PSI-style scoring.
- `logging_utils.py`: House-specific structured JSON and CloudWatch EMF output.
- `config.py`: optional House configuration loaded only when that product is
  selected.

Iris logging and model-loading modules are not refactored as part of this work.
Shared S3 access may be reused through an additive generic interface, but the
existing Iris validation path remains untouched.

### Streamlit application

`app.py` gains product navigation with Iris as the default. The House page is
rendered only after selection. Its configuration and `st.cache_resource` model
loader are called inside the House branch, so missing House environment
variables or artifacts cannot affect the Iris branch.

The House page contains:

- required numeric and categorical inputs;
- model, dataset, and schema versions;
- training metrics and educational explanation;
- one-property prediction and safe structured logging;
- CSV upload with extension, size, row-count, schema, and value validation;
- preview and download of prediction results;
- optional-label evaluation metrics;
- batch drift diagnostics and educational threshold warnings.

### Training and intake scripts

- `scripts/intake_house_price_dataset.py` downloads or accepts a local Kaggle
  archive, extracts only the expected `train.csv`, validates it, calculates
  provenance, and prepares an uploadable versioned object.
- `scripts/train_house_price_model.py` accepts a specific local dataset or
  exact S3 object version, refuses missing/invalid real input, trains candidates,
  logs MLflow, creates the complete bundle, and reports the selected model and
  metrics.
- No script writes Kaggle credentials, datasets, or model artifacts into Git.

The production S3 prefixes are:

```text
datasets/house-prices/v1/train.csv
models/house-price/v1/model.tar.gz
mlflow/
```

The application receives the exact model key and S3 `VersionId`; it never
resolves a mutable `latest` object.

## Training design

Training uses a deterministic two-stage split to produce train, validation, and
test partitions with a fixed seed. Preprocessing is fitted only on the training
partition. Candidate models are:

- `DummyRegressor` baseline;
- CPU-friendly `RandomForestRegressor`;
- CPU-friendly `GradientBoostingRegressor`.

The selected candidate is the one with the best validation RMSE in original
price units, with deterministic tie-breaking by declared candidate order. The
test split is evaluated once after selection and remains untouched by
preprocessing fitting, model fitting, and selection.

Tracked MLflow metadata includes dataset version/checksum, Git SHA, ordered
features and types, preprocessing configuration, model type/parameters, all
four metrics, duration, and random seed. Local development uses a file-backed
MLflow URI. The AWS training workflow uses the existing Region and a managed
SageMaker AI MLflow Tracking Server when one is available; if none exists, the
workflow provisions at most one project-scoped server after checking the
existing budget and documents its ARN, IAM permissions, and costs.

Model registry names and aliases are explicit:

```text
house-price-model
candidate
champion
```

Promotion is a recorded workflow decision requiring explicit approval. No
training run silently promotes a model.

## Bundle and integrity design

Every model bundle contains:

```text
model.tar.gz
├── model.joblib
├── manifest.json
├── metrics.json
├── reference_profile.json
├── feature_schema.json
└── environment.json
```

The manifest records model/dataset versions, source URL, dataset checksum, Git
SHA, feature names/types, target, model type, framework versions, training
parameters, metrics, training timestamp, member checksums, and a canonical
bundle checksum.

Because a manifest containing its own outer archive checksum would be circular,
the bundle checksum is defined over a canonical sorted member stream with the
manifest checksum field normalized before hashing. The loader reproduces that
canonicalization before deserialization. All archive paths, member sizes,
member names, and checksums are validated before `joblib.load` is called.

## Drift and observability

Training writes a reference profile containing numeric bounds, mean, standard
deviation, quantile/bin information, and categorical distributions. Batch
prediction calculates a documented educational PSI-style score per feature.
The default warning threshold is `0.20`; it is explicitly not presented as a
universal production standard. A one-row online prediction does not claim to
measure drift.

House structured logs include product, model version, dataset version, Git SHA,
Docker image version, safe counts, latency, and validation state. Raw rows,
addresses, property identifiers, and full feature inputs are never logged.

House EMF metrics use only `Environment`, `Product`, and `ModelVersion`
dimensions:

```text
HousePredictionCount
HousePredictionErrorCount
HousePredictionLatencyMs
HouseBatchPredictionCount
HouseBatchRowsProcessed
HouseInvalidInputCount
HouseDriftedFeatureCount
HouseModelLoadFailure
HouseBatchMAE
HouseBatchRMSE
```

Iris metrics and alarms remain unchanged. House metrics use a separate namespace
or metric names that cannot collide with existing Iris alarms.

## Security and limits

- Kaggle credentials are read only from external environment/configuration.
- Uploaded files must be CSV, have a bounded size, and stay below a configured
  maximum row count.
- Unexpected columns, malformed CSV, invalid values, and missing required
  fields receive stable non-sensitive errors.
- S3 remains private and versioned.
- ECS reads only the House model prefix and existing Iris model object.
- GitHub OIDC trust remains repository- and branch-restricted.
- No request IDs, property IDs, addresses, or raw inputs are metric dimensions.
- Serialized objects are trusted only after checksum and provenance validation.

## Testing strategy

All tests run without Kaggle or live AWS access. A small committed fixture is
used only for tests and is clearly labeled as a fixture, not a production
dataset or production result.

Tests cover:

- raw schema and target validation;
- feature type detection and split reproducibility;
- preprocessing leakage boundaries;
- missing and unknown values;
- candidate training and metric calculations;
- bundle creation, checksums, and trusted loading;
- prediction and batch output schemas;
- CSV limits and validation errors;
- reference profiles and drift scores;
- structured events and EMF dimensions;
- Streamlit House and Iris regression behavior;
- Docker startup and health endpoint;
- PR, training, and deployment workflow contracts.

Focused tests are run before the full existing Iris suite. Baseline failures are
reported separately from changes introduced by House Pricing.

## Deployment and rollback

The deployment workflow builds a commit-tagged immutable image, passes existing
Iris coordinates plus exact House model coordinates, updates the existing ECS
Express service, waits for a successful deployment, and checks the health
endpoint. Post-deployment verification covers both pages, valid/invalid batch
CSV behavior, CloudWatch logs, and House metrics.

Rollback means redeploying the previous immutable image and restoring the
previous exact House model key/VersionId. Retraining is manual or dataset/code
change-triggered, never coupled to every application deployment.

## Explicit non-goals

- No second public service.
- No full 79-column online schema.
- No automatic production promotion.
- No automated Kaggle retraining on every deploy.
- No committed dataset, credential, model binary, or mutable latest artifact.
- No claims that Ames/Iowa historical data generalizes to other geographies or
  current markets.

## Delivery handoff

The design is approved for implementation on `feat/house-pricing-mlops`. The
next step is a test-first implementation plan, followed by incremental code,
fixture tests, real Kaggle intake, and verified deployment where the real
dataset and AWS training prerequisites are available.
