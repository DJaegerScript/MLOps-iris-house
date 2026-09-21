"""Start the House-only application pipeline with one approved release."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from house_pricing_mlops.release import HouseReleaseManifest, ReleaseValidationError
from scripts.codepipeline.validate_house_release import load_release

HOUSE_APP_PIPELINE_NAME = "house-pricing-app"
HOUSE_DEPLOYMENT_SERVICE_NAME = "house-pricing"


def load_approved_release(path: str | Path) -> HouseReleaseManifest:
    """Load a champion release artifact suitable for deployment."""

    return load_release(path, require_champion=True)


def resolve_release_from_ssm(
    ssm_client: Any, *, parameter_name: str
) -> HouseReleaseManifest:
    """Resolve and validate the current approved release pointer once."""

    if not parameter_name.strip():
        raise ReleaseValidationError("approved release parameter name is required")
    try:
        response = ssm_client.get_parameter(
            Name=parameter_name,
            WithDecryption=False,
        )
        value = response["Parameter"]["Value"]
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseValidationError(
            "approved release parameter is malformed"
        ) from error
    release = HouseReleaseManifest.from_json(value)
    if release.status != "champion":
        raise ReleaseValidationError("approved release parameter is not champion")
    return release


def release_pipeline_variables(
    release: HouseReleaseManifest,
) -> list[dict[str, str]]:
    """Return exact release coordinates for CodePipeline variables."""

    if release.status != "champion":
        raise ReleaseValidationError("only a champion release can start deployment")
    return [
        {"name": "HOUSE_MODEL_S3_BUCKET", "value": release.model_s3_bucket},
        {"name": "HOUSE_MODEL_S3_KEY", "value": release.model_s3_key},
        {
            "name": "HOUSE_MODEL_S3_VERSION_ID",
            "value": release.model_s3_version_id,
        },
        {"name": "HOUSE_MODEL_VERSION", "value": release.model_version},
        {"name": "HOUSE_DATASET_VERSION", "value": release.dataset_version},
    ]


def start_app_pipeline(
    codepipeline_client: Any,
    *,
    pipeline_name: str,
    release: HouseReleaseManifest,
    extra_variables: Mapping[str, str] | None = None,
) -> str:
    """Start the app pipeline with no mutable model reference."""

    if not pipeline_name.strip():
        raise ReleaseValidationError("application pipeline name is required")
    variables = release_pipeline_variables(release)
    for name, value in (extra_variables or {}).items():
        if not isinstance(name, str) or not name.strip():
            raise ReleaseValidationError("pipeline variable names are required")
        if not isinstance(value, str):
            raise ReleaseValidationError(f"pipeline variable {name} must be text")
        variables.append({"name": name, "value": value})
    response = codepipeline_client.start_pipeline_execution(
        name=pipeline_name,
        variables=variables,
    )
    try:
        return str(response["pipelineExecutionId"])
    except (KeyError, TypeError) as error:
        raise ReleaseValidationError(
            "CodePipeline did not return an execution id"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-name", default=HOUSE_APP_PIPELINE_NAME)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--approved-release-parameter-name")
    parser.add_argument("--region")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import boto3

        if args.release is not None:
            release = load_approved_release(args.release)
        elif args.approved_release_parameter_name:
            release = resolve_release_from_ssm(
                boto3.client("ssm", region_name=args.region),
                parameter_name=args.approved_release_parameter_name,
            )
        else:
            raise ReleaseValidationError(
                "provide --release or --approved-release-parameter-name"
            )
        execution_id = start_app_pipeline(
            boto3.client("codepipeline", region_name=args.region),
            pipeline_name=args.pipeline_name,
            release=release,
        )
    except (ReleaseValidationError, OSError, ValueError) as error:
        print(f"application pipeline handoff failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "pipeline_name": args.pipeline_name,
                "deployment_service": HOUSE_DEPLOYMENT_SERVICE_NAME,
                "pipeline_execution_id": execution_id,
                "house_model_version": release.model_version,
                "house_model_s3_version_id": release.model_s3_version_id,
                "status": "STARTED",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
