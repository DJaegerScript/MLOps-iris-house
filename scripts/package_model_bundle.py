#!/usr/bin/env python3
"""Validate and package the approved Iris v1 artifacts for operator upload.

This script is an offline intake step. It reads four artifacts that an operator
has already obtained from the immutable public source, verifies their exact
checksums, and writes a deterministic tar.gz bundle. It never downloads model
files, accesses AWS, deserializes pickle files, or belongs in the application
startup path.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import sys
import tarfile
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iris_mlops.manifest import V1_EXPECTATIONS

MAX_FILE_BYTES: Final = 16 * 1024 * 1024
ARTIFACT_NAMES: Final = tuple(V1_EXPECTATIONS.artifact_checksums)


class IntakeValidationError(ValueError):
    """Raised when the offline intake directory is not the approved bundle."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_intake_directory(input_dir: Path) -> dict[str, Path]:
    """Validate exactly the approved v1 files without loading their contents."""

    if not input_dir.is_dir():
        raise IntakeValidationError(f"intake directory does not exist: {input_dir}")

    files = {
        path.name: path
        for path in input_dir.iterdir()
        if path.is_file()
    }
    unexpected = sorted(set(files) - set(ARTIFACT_NAMES))
    if unexpected:
        raise IntakeValidationError(
            "unexpected intake files: " + ", ".join(unexpected)
        )

    validated: dict[str, Path] = {}
    for name in ARTIFACT_NAMES:
        path = files.get(name)
        if path is None:
            raise IntakeValidationError(f"missing intake artifact: {name}")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise IntakeValidationError(
                f"intake artifact exceeds {MAX_FILE_BYTES} bytes: {name}"
            )
        actual = _sha256(path)
        expected = V1_EXPECTATIONS.artifact_checksums[name]
        if actual != expected:
            raise IntakeValidationError(
                f"checksum mismatch for intake artifact: {name}"
            )
        validated[name] = path
    return validated


def package_v1_bundle(
    input_dir: Path, output_path: Path, *, force: bool = False
) -> str:
    """Create the deterministic, checksum-pinned v1 bundle and return its SHA-256."""

    files = validate_intake_directory(input_dir)
    if output_path.exists() and not force:
        raise IntakeValidationError(
            f"output already exists; pass --force to replace it: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("wb") as raw:
            with gzip.GzipFile(
                fileobj=raw, mode="wb", filename="", mtime=0
            ) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as archive:
                    for name in ARTIFACT_NAMES:
                        data = files[name].read_bytes()
                        info = tarfile.TarInfo(name=name)
                        info.size = len(data)
                        info.mode = 0o644
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        archive.addfile(info, io.BytesIO(data))

        bundle_sha256 = _sha256(temporary_path)
        expected_bundle_sha256 = V1_EXPECTATIONS.bundle_sha256
        if bundle_sha256 != expected_bundle_sha256:
            raise IntakeValidationError(
                "packaged bundle checksum does not match the approved v1 checksum"
            )
        if output_path.exists() and force:
            output_path.unlink()
        os.replace(temporary_path, output_path)
        return bundle_sha256
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    checksum = package_v1_bundle(args.input_dir, args.output, force=args.force)
    print(f"packaged={args.output}")
    print(f"sha256={checksum}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
