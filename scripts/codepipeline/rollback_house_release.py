"""Start an exact-coordinate House application rollback."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from house_pricing_mlops.release import HouseReleaseManifest, ReleaseValidationError
from scripts.codepipeline.invoke_house_app_pipeline import (
    load_approved_release,
    start_app_pipeline,
)

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def validate_rollback_arguments(
    *,
    previous_image_uri: str,
    previous_image_digest: str,
    previous_release: HouseReleaseManifest,
    reason: str,
) -> dict[str, str]:
    """Validate rollback coordinates without changing any stored artifact."""

    if not previous_image_uri.strip() or "latest" in previous_image_uri.lower():
        raise ValueError("previous image must be an immutable image reference")
    if not _DIGEST_PATTERN.fullmatch(previous_image_digest.strip()):
        raise ValueError("previous image digest must be a sha256 digest")
    if previous_release.status != "champion":
        raise ReleaseValidationError("previous release must be champion")
    if not reason.strip():
        raise ValueError("rollback reason is required")
    return {
        "image_uri": previous_image_uri.strip(),
        "image_digest": previous_image_digest.strip(),
        "house_model_s3_bucket": previous_release.model_s3_bucket,
        "house_model_s3_key": previous_release.model_s3_key,
        "house_model_s3_version_id": previous_release.model_s3_version_id,
        "house_model_version": previous_release.model_version,
        "house_dataset_version": previous_release.dataset_version,
        "reason": reason.strip(),
        "status": "ROLLBACK_REQUESTED",
    }


def start_rollback(
    codepipeline_client: Any,
    *,
    pipeline_name: str,
    previous_image_uri: str,
    previous_image_digest: str,
    previous_release: HouseReleaseManifest,
    reason: str,
) -> tuple[str, dict[str, str]]:
    coordinates = validate_rollback_arguments(
        previous_image_uri=previous_image_uri,
        previous_image_digest=previous_image_digest,
        previous_release=previous_release,
        reason=reason,
    )
    execution_id = start_app_pipeline(
        codepipeline_client,
        pipeline_name=pipeline_name,
        release=previous_release,
        extra_variables={
            "IMAGE_URI_OVERRIDE": coordinates["image_uri"],
            "IMAGE_DIGEST_OVERRIDE": coordinates["image_digest"],
            "ROLLBACK_REASON": coordinates["reason"],
        },
    )
    return execution_id, coordinates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-name", default="house-pricing-app")
    parser.add_argument("--previous-image-uri", required=True)
    parser.add_argument("--previous-image-digest", required=True)
    parser.add_argument("--previous-release", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--region", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import boto3

        release = load_approved_release(args.previous_release)
        execution_id, coordinates = start_rollback(
            boto3.client("codepipeline", region_name=args.region),
            pipeline_name=args.pipeline_name,
            previous_image_uri=args.previous_image_uri,
            previous_image_digest=args.previous_image_digest,
            previous_release=release,
            reason=args.reason,
        )
    except (OSError, ValueError, ReleaseValidationError) as error:
        print(f"House rollback failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "pipeline_name": args.pipeline_name,
                "pipeline_execution_id": execution_id,
                **coordinates,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
