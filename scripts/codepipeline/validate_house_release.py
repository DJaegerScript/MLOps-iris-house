"""Validate one finalized House release artifact for pipeline handoff."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from house_pricing_mlops.release import HouseReleaseManifest, ReleaseValidationError


def load_release(
    path: str | Path, *, require_champion: bool = False
) -> HouseReleaseManifest:
    """Load a release and optionally require champion promotion status."""

    try:
        manifest = HouseReleaseManifest.from_json(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, ReleaseValidationError) as error:
        raise ReleaseValidationError(f"release artifact is invalid: {error}") from error
    if require_champion and manifest.status != "champion":
        raise ReleaseValidationError(
            "release artifact is not a promoted champion with an exact S3 VersionId"
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--require-champion", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_release(args.path, require_champion=args.require_champion)
    except ReleaseValidationError as error:
        print(f"release validation failed: {error}", file=sys.stderr)
        return 2
    print(manifest.to_json())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
