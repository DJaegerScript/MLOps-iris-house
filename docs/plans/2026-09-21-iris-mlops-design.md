# Iris MLOps Showcase Design

**Date:** 2026-09-21

**Scope:** Iris classification deployment only. House Pricing is intentionally excluded.

## Goal

Demonstrate model artifact intake, validation, packaging, containerization, deployment, prediction logging, and CloudWatch monitoring for a pre-trained Iris classifier served by Streamlit.

## Storage and trust boundaries

The inference model bundle is stored only in a private, versioned S3 bucket. It is not committed to GitHub, copied into the Docker image, or included in test fixtures. The repository stores only non-secret artifact metadata: source URL, immutable source revision, checksums, feature schema, class names, framework information, and model version.

No application secret is required for the first version. AWS deployment uses GitHub Actions OIDC and IAM roles rather than static credentials. If a future application secret is required, it must be stored as an encrypted private S3 object and read only by a narrowly scoped ECS task role; it must never be committed, baked into the image, passed through source-controlled configuration, or written to logs.

## Artifact flow

1. Inspect a vetted public educational Iris model source and pin an immutable revision.
2. Download the model and any required scaler or encoder to a temporary intake location.
3. Inspect the source documentation and serialized object structure before loading it.
4. Validate the four-feature schema, three expected classes, probability support, Python/scikit-learn metadata, and source-verified training information.
5. Calculate SHA-256 checksums for every bundle member and the bundle archive.
6. Upload the bundle and manifest to the private versioned S3 bucket.
7. ECS downloads the immutable S3 object using its task role.
8. The application verifies the checksum before deserializing and caches the loaded artifacts once.

The application never trains, retrains, or downloads an unverified artifact during startup.

## Application architecture

- `app.py`: Streamlit UI on `0.0.0.0:8501`.
- `src/iris_mlops/manifest.py`: strict manifest and feature-schema validation.
- `src/iris_mlops/model.py`: S3 download, checksum verification, artifact loading, class/probability validation, and prediction.
- `src/iris_mlops/logging_utils.py`: structured JSON stdout logs and CloudWatch Embedded Metric Format.
- `src/iris_mlops/config.py`: environment-driven runtime configuration without secrets in source.
- `tests/`: unit tests using mocked storage boundaries plus deployment-time tests against the real S3 artifact.

The UI has four numeric inputs, clear validation errors, a prediction button, predicted species, class probabilities when supported, model version/provenance, and a short educational Iris explanation.

## Observability

Each prediction emits structured JSON with event, model version, predicted class, confidence, latency, and validity. Startup, model-load success/failure, invalid input, prediction errors, Git commit SHA, and Docker image version are also logged.

EMF metrics use only `Environment` and `ModelVersion` dimensions:

- `PredictionCount`
- `PredictionErrorCount`
- `PredictionLatencyMs`
- `LowConfidencePredictionCount`
- `InvalidInputCount`
- `ModelLoadFailure`

## AWS architecture

The target region is `ap-southeast-3` in account `163918295215`. Resources use the `iris-mlops` prefix and are created only if absent:

- Private versioned S3 model bucket with public access blocked.
- Private ECR repository.
- ECS Express Mode service on Fargate.
- CloudWatch log group with retention.
- ECS task execution role.
- ECS Express infrastructure role.
- ECS task role restricted to the Iris model S3 prefix.
- GitHub Actions OIDC deployment role restricted to `DJaegerScript/MLOps-iris-house` and `main`.
- CloudWatch alarms for application errors, model-load failures, and load-balancer/service health.
- Billing budget alert, subject to a notification address being available.

The public application URL will be the ECS Express Mode HTTPS URL. The service will use the Streamlit health endpoint `/_stcore/health`.

## CI/CD

Pull requests install pinned dependencies, validate metadata, run tests, run linting, build the application-only Docker image, and run a Docker smoke test. Deployment authenticates through OIDC, builds an image tagged with the commit SHA, pushes it to ECR, updates ECS Express Mode, and reports the image and S3 model version.

No workflow contains a model-training step.

## Verification and operations

After deployment, verify the public URL, health endpoint, live prediction, CloudWatch logs, EMF metrics, resource names and ARNs, deployed image tag, and deployed model version. README documentation covers local setup, Docker, AWS prerequisites, deployment, rollback, cleanup, provenance, limitations, resource scope, and cost considerations.

