"""Run safe public and exact-coordinate House release smoke checks.

The command reports release coordinates, statuses, counts, and latency only.
It never prints uploaded rows, addresses, identifiers, credentials, or model
predictions. The optional local model check reads one immutable S3 object
version and exercises the same validation services used by the application.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from house_pricing_mlops.config import HouseSettings
from house_pricing_mlops.prediction import (
    HouseBatchPredictionService,
    HouseBatchValidationError,
    HousePredictionService,
    load_model_from_s3,
)
from house_pricing_mlops.release import HouseReleaseManifest, ReleaseValidationError
from house_pricing_mlops.schema import MODEL_FEATURES, TARGET_NAME
from iris_mlops.storage import S3ArtifactStore
from scripts.codepipeline.validate_house_release import load_release

HEALTH_PATH = "/_stcore/health"
_IMAGE_DIGEST_PREFIX = "sha256:"


class _SilentLogger:
    """Logger adapter that deliberately discards smoke-test event values."""

    def invalid_input(self) -> None:
        pass

    def prediction_success(self, **_: object) -> None:
        pass

    def prediction_error(self, **_: object) -> None:
        pass

    def batch_validation_error(self, **_: object) -> None:
        pass

    def batch_prediction_error(self, **_: object) -> None:
        pass

    def batch_prediction(self, **_: object) -> None:
        pass


def validate_release_coordinates(
    image_release: Mapping[str, object], release: HouseReleaseManifest
) -> dict[str, str]:
    """Validate and return only immutable image/model release coordinates."""

    if release.status != "champion":
        raise ReleaseValidationError("smoke test requires a champion release")
    image_uri = _required_text(image_release, "image_uri")
    commit_sha = _required_text(image_release, "commit_sha")
    digest = _required_text(image_release, "digest")
    if "latest" in image_uri.lower():
        raise ValueError("image URI cannot use latest")
    if not digest.startswith(_IMAGE_DIGEST_PREFIX) or len(digest) != 71:
        raise ValueError("image digest must be an immutable sha256 digest")
    if any(character not in "0123456789abcdef" for character in digest[7:]):
        raise ValueError("image digest must be an immutable sha256 digest")
    return {
        "image_uri": image_uri,
        "image_commit_sha": commit_sha,
        "image_digest": digest,
        "house_model_s3_bucket": release.model_s3_bucket,
        "house_model_s3_key": release.model_s3_key,
        "house_model_s3_version_id": release.model_s3_version_id,
        "house_model_version": release.model_version,
        "house_dataset_version": release.dataset_version,
        "house_bundle_sha256": release.bundle_sha256,
    }


def build_safe_batch_csv(*, labeled: bool) -> bytes:
    """Build a deterministic, non-identifying one-row smoke-test CSV."""

    row: dict[str, object] = {
        "OverallQual": 7,
        "GrLivArea": 1710,
        "GarageCars": 2,
        "TotalBsmtSF": 856,
        "1stFlrSF": 856,
        "YearBuilt": 2003,
        "FullBath": 2,
        "TotRmsAbvGrd": 8,
        "GarageArea": 548,
        "Neighborhood": "CollgCr",
        "KitchenQual": "Gd",
        "CentralAir": "Y",
    }
    fieldnames = list(MODEL_FEATURES)
    if labeled:
        row[TARGET_NAME] = 208500
        fieldnames.append(TARGET_NAME)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    return output.getvalue().encode("utf-8")


def build_smoke_summary(
    image_release: Mapping[str, object],
    release: HouseReleaseManifest,
    *,
    health_status: str,
    latency_ms: float,
    checks: Mapping[str, str],
) -> dict[str, object]:
    """Build a safe operator report without request or response payloads."""

    coordinates = validate_release_coordinates(image_release, release)
    passed = health_status == "ok" and all(
        value == "passed" for value in checks.values()
    )
    return {
        "status": "PASSED" if passed else "FAILED",
        "health_status": health_status,
        "latency_ms": round(float(latency_ms), 3),
        "checks": dict(sorted(checks.items())),
        **coordinates,
    }


def check_public_health(
    public_url: str, *, timeout_seconds: float = 10.0
) -> tuple[str, float]:
    """Check only the public Streamlit health endpoint and measure latency."""

    url = public_url.rstrip("/") + HEALTH_PATH
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            body = response.read(32).decode("utf-8", errors="replace").strip()
            if response.status != 200 or body != "ok":
                raise RuntimeError("public health endpoint did not return ok")
    except (OSError, urllib.error.URLError) as error:
        raise RuntimeError("public health endpoint is unavailable") from error
    return "ok", (time.perf_counter() - started) * 1000


def run_exact_model_checks(
    release: HouseReleaseManifest, *, region: str | None = None
) -> dict[str, str]:
    """Exercise online, labeled, unlabeled, and invalid CSV contracts safely."""

    import boto3

    model = load_model_from_s3(
        S3ArtifactStore(client=boto3.client("s3", region_name=region)),
        release.model_s3_bucket,
        release.model_s3_key,
        release.model_s3_version_id,
    )
    settings = HouseSettings(
        model_s3_bucket=release.model_s3_bucket,
        model_s3_key=release.model_s3_key,
        model_s3_version_id=release.model_s3_version_id,
        model_version=release.model_version,
        dataset_version=release.dataset_version,
        environment="smoke-test",
        git_commit_sha=release.git_commit_sha,
        docker_image_version="smoke-test",
    )
    logger = _SilentLogger()
    profile = model.reference_profile
    row = _safe_features_from_profile(profile)
    HousePredictionService(model, settings, logger=logger).predict(row)
    batch = HouseBatchPredictionService(
        model,
        settings,
        reference_profile=profile,
        logger=logger,
    )
    batch.predict_csv(build_safe_batch_csv(labeled=True), filename="labeled.csv")
    batch.predict_csv(build_safe_batch_csv(labeled=False), filename="unlabeled.csv")
    try:
        batch.predict_csv(
            b"not,a,valid,house,batch\n1,2,3,4,5\n",
            filename="invalid.csv",
        )
    except HouseBatchValidationError:
        pass
    else:
        raise RuntimeError("invalid batch CSV was accepted")
    return {
        "iris_sample": "not_run_by_this_harness",
        "house_online": "passed",
        "house_batch_labeled": "passed",
        "house_batch_unlabeled": "passed",
        "house_batch_invalid": "passed",
    }


def _safe_features_from_profile(profile: Mapping[str, Any]) -> dict[str, object]:
    features: dict[str, object] = {}
    for name, values in profile["numeric"].items():
        value = float(values["mean"])
        if name == "OverallQual":
            value = min(10.0, max(1.0, value))
        elif name == "YearBuilt":
            value = min(2100.0, max(1800.0, value))
        else:
            value = max(0.0, value)
        features[name] = value
    for name, values in profile["categorical"].items():
        categories = values.get("categories", [])
        if not categories:
            raise RuntimeError(f"reference profile has no category for {name}")
        features[name] = str(categories[0])
    return features


def _required_text(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-url", required=True)
    parser.add_argument("--approved-release", type=Path, required=True)
    parser.add_argument("--image-release", type=Path, required=True)
    parser.add_argument("--region", default=None)
    parser.add_argument("--skip-model", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        release = load_release(args.approved_release, require_champion=True)
        image_release = json.loads(args.image_release.read_text(encoding="utf-8"))
        validate_release_coordinates(image_release, release)
        health_status, latency_ms = check_public_health(args.public_url)
        checks = {
            "public_health": "passed",
            "iris_sample": "manual_ui_check_required",
        }
        if args.skip_model:
            checks.update({
                "house_online": "skipped",
                "house_batch_labeled": "skipped",
                "house_batch_unlabeled": "skipped",
                "house_batch_invalid": "skipped",
            })
        else:
            checks.update(run_exact_model_checks(release, region=args.region))
        summary = build_smoke_summary(
            image_release,
            release,
            health_status=health_status,
            latency_ms=latency_ms,
            checks=checks,
        )
        print(json.dumps(summary, sort_keys=True))
    except (OSError, ValueError, ReleaseValidationError, RuntimeError) as error:
        print(f"House smoke test failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
