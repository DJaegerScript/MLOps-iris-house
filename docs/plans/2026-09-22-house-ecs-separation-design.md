# House Pricing ECS Deployment Separation Design

## Goal

Keep the existing `iris-mlops` ECS Express service running the Iris use case
without pipeline changes, while deploying the House Pricing use case through
the AWS CodePipeline workflow to a separate ECS Express service.

## Approved architecture

- The GitHub repository remains shared.
- The existing `iris-mlops` ECS service, endpoint, task role, execution role,
  ECR repository, and log group remain unchanged by the House pipeline.
- The application gains an explicit runtime variant. The existing/default
  variant is Iris; the new `house` variant renders only House Pricing and does
  not load Iris configuration or the Iris model.
- A new ECS Express service named `house-pricing` runs the `house` variant in
  the existing `iris-mlops` cluster. It has a separate endpoint and log group.
- The House service uses a dedicated task role with only the exact versioned
  House model S3 object permissions. The existing Iris task role is not reused
  for the new service.
- The existing ECR repository and ECS execution role are reused because they
  are image/runtime infrastructure, not use-case state. The image is built
  once from the shared repository; the runtime variant determines which product
  is served.
- `house-pricing-app` deploys only the new House service. Its first deployment
  creates the service if absent; later deployments update it and wait for the
  new ECS Express deployment to become successful.
- House smoke tests use the new House endpoint and verify that the deployment
  is pinned to the promoted image digest and model VersionId.

## Data flow

1. GitHub `main` feeds the existing regional GitHub webhook into
   `house-pricing-app`.
2. CodeBuild builds/tests the shared image and publishes immutable image
   metadata.
3. The model pipeline promotes a House release and starts the app pipeline
   with its exact model coordinates.
4. The app deploy stage builds a primary-container definition with
   `APP_VARIANT=house`, the image digest, and the approved House model
   VersionId.
5. The deploy stage creates or updates only `house-pricing`, waits for ECS
   Express success, and checks the House health endpoint.

## Failure and rollback boundaries

- A House build or deployment failure leaves the existing Iris service
  untouched.
- House rollback starts the House app pipeline with a prior immutable image
  digest and prior champion model coordinates; it never targets `iris-mlops`.
- The deploy role is scoped to the House service, House task role, House log
  group, and required exact model/artifact paths.

## Verification

- Unit/page tests prove the default Iris variant and House-only variant render
  the intended product boundary.
- Infrastructure tests prove the House service ARN, dedicated task role/log
  group, `APP_VARIANT=house`, and absence of the existing Iris service ARN from
  House deployment commands.
- Full local tests, Ruff, YAML/CloudFormation validation, and `git diff --check`
  pass before AWS mutation.
- Live verification records the existing Iris service configuration before and
  after deployment, then checks the new House endpoint, image digest, model
  VersionId, health, and House prediction smoke path.
