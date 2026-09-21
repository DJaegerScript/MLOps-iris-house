#!/usr/bin/env bash
set -euo pipefail

service_arn="${1:?service ARN is required}"
previous_deployment="${2:-}"
max_attempts="${MAX_ATTEMPTS:-40}"
sleep_seconds="${SLEEP_SECONDS:-15}"

for ((attempt = 1; attempt <= max_attempts; attempt += 1)); do
  deployment_arn="$(aws ecs list-service-deployments \
    --service "$service_arn" \
    --query 'serviceDeployments[0].serviceDeploymentArn' \
    --output text)"
  if [[ -z "$deployment_arn" || "$deployment_arn" == "None" || "$deployment_arn" == "$previous_deployment" ]]; then
    if (( attempt == max_attempts )); then
      echo "timeout waiting for a new ECS Express deployment" >&2
      exit 1
    fi
    sleep "$sleep_seconds"
    continue
  fi

  status="$(aws ecs describe-service-deployments \
    --service-deployment-arns "$deployment_arn" \
    --query 'serviceDeployments[0].status' \
    --output text)"
  case "$status" in
    SUCCESSFUL)
      echo "$deployment_arn"
      exit 0
      ;;
    FAILED|STOPPED)
      echo "ECS Express deployment $deployment_arn ended with status $status" >&2
      exit 1
      ;;
  esac
  if (( attempt == max_attempts )); then
    echo "timeout waiting for ECS Express deployment $deployment_arn (status $status)" >&2
    exit 1
  fi
  sleep "$sleep_seconds"
done
