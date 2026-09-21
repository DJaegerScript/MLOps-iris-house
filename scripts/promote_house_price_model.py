"""Register, promote, or roll back a House Pricing model record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from house_pricing_mlops.registry import RegistryError, VersionedModelRegistry


def register_from_summary(
    registry: VersionedModelRegistry,
    summary_path: str | Path,
    *,
    s3_version_id: str | None = None,
) -> dict[str, object]:
    """Register one training summary as a candidate without promotion."""

    try:
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryError("training summary is unreadable") from error
    if not isinstance(summary, dict):
        raise RegistryError("training summary must contain an object")
    s3_object = summary.get("s3_object")
    if not isinstance(s3_object, dict):
        raise RegistryError("training summary is missing S3 object metadata")
    dataset_s3_bucket = summary.get("dataset_s3_bucket")
    dataset_s3_key = summary.get("dataset_s3_key")
    dataset_s3_version_id = summary.get("dataset_s3_version_id")
    model_s3_bucket = summary.get("model_s3_bucket")
    model_s3_key = summary.get("model_s3_key")
    git_commit_sha = summary.get("git_commit_sha")
    if all(
        isinstance(value, str) and value.strip()
        for value in (
            dataset_s3_bucket,
            dataset_s3_key,
            dataset_s3_version_id,
            model_s3_bucket,
            model_s3_key,
            git_commit_sha,
        )
    ):
        manifest = registry.register_candidate_release(
            model_name=str(summary.get("model_name", "")),
            model_version=str(summary.get("model_version", "")),
            run_id=str(summary.get("mlflow_run_id", "")),
            dataset_version=str(summary.get("dataset_version", "")),
            dataset_s3_bucket=dataset_s3_bucket,
            dataset_s3_key=dataset_s3_key,
            dataset_s3_version_id=dataset_s3_version_id,
            bundle_sha256=str(summary.get("bundle_sha256", "")),
            model_s3_bucket=model_s3_bucket,
            model_s3_key=model_s3_key,
            model_s3_version_id=s3_version_id or str(s3_object.get("version_id", "")),
            selected_model=str(summary.get("selected_model", "")),
            metrics=summary.get("metrics", {})
            if isinstance(summary.get("metrics"), dict)
            else {},
            git_commit_sha=git_commit_sha,
        )
        return manifest.to_dict()
    return registry.register_candidate(
        model_name=str(summary.get("model_name", "")),
        model_version=str(summary.get("model_version", "")),
        run_id=str(summary.get("mlflow_run_id", "")),
        dataset_version=str(summary.get("dataset_version", "")),
        dataset_sha256=str(summary.get("dataset_sha256", "")),
        bundle_sha256=str(summary.get("bundle_sha256", "")),
        s3_bucket=str(s3_object.get("bucket", "")),
        s3_key=str(s3_object.get("key", "")),
        s3_version_id=s3_version_id or str(s3_object.get("version_id", "")),
        selected_model=str(summary.get("selected_model", "")),
        metrics=summary.get("metrics", {})
        if isinstance(summary.get("metrics"), dict)
        else {},
    )


def promote_release(
    registry: VersionedModelRegistry,
    registry_version: str,
    *,
    approved_by: str,
    reason: str,
) -> dict[str, object]:
    """Promote one candidate and persist its approval metadata."""

    return registry.promote_champion(
        registry_version,
        approved_by=approved_by,
        reason=reason,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register-candidate")
    register.add_argument("--summary", type=Path, required=True)
    register.add_argument("--s3-version-id")

    for command in ("promote", "rollback"):
        action = subparsers.add_parser(command)
        action.add_argument("--registry-version", required=True)
        action.add_argument("--approved-by", required=True)
        action.add_argument("--reason", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = VersionedModelRegistry(args.registry)
    try:
        if args.command == "register-candidate":
            result = register_from_summary(
                registry,
                args.summary,
                s3_version_id=args.s3_version_id,
            )
        elif args.command == "promote":
            result = promote_release(
                registry,
                args.registry_version,
                approved_by=args.approved_by,
                reason=args.reason,
            )
        else:
            result = registry.rollback_champion(
                args.registry_version,
                approved_by=args.approved_by,
                reason=args.reason,
            )
    except RegistryError as error:
        print(f"model registry operation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
