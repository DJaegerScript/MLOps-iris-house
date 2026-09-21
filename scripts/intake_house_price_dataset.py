"""Validate a downloaded Kaggle House Prices input and write provenance JSON.

This command deliberately stops at the local intake boundary. It does not
download from Kaggle, upload to S3, or print credentials. Operators can use
the Kaggle CLI separately, for example:

    kaggle competitions download \
      -c house-prices-advanced-regression-techniques \
      -p /tmp/house-prices
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd

from house_pricing_mlops.provenance import (
    DEFAULT_DATASET_SOURCE_URL,
    build_dataset_provenance,
)
from house_pricing_mlops.schema import SchemaValidationError

FIXTURE_SOURCE_URL = "fixture://house-prices/test-only"
DEFAULT_S3_KEY_TEMPLATE = "datasets/house-prices/{dataset_version}/train.csv"
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class IntakeError(ValueError):
    """A safe and actionable local intake failure."""


def intake_dataset(
    input_path: str | Path,
    *,
    dataset_version: str,
    output_dir: str | Path,
    source_url: str | None = None,
    s3_bucket: str | None = None,
    s3_key: str | None = None,
    download_timestamp: str | None = None,
    git_commit_sha: str | None = None,
) -> dict[str, object]:
    """Validate ``train.csv`` and write a JSON-only intake record.

    A ZIP is read without extracting arbitrary members to disk. The output
    directory receives only ``provenance.json``; the production dataset must
    be uploaded separately to the returned exact S3 key after operator review.
    """

    _validate_dataset_version(dataset_version)
    input_file = Path(input_path)
    if not input_file.is_file():
        raise IntakeError("dataset input does not exist")

    effective_source = source_url or _source_url_for_input(input_file)
    with _materialize_train_csv(input_file) as train_csv:
        try:
            _validate_csv_header(train_csv)
            provenance = build_dataset_provenance(
                train_csv,
                dataset_version=dataset_version,
                source_url=effective_source,
                download_timestamp=download_timestamp,
                git_commit_sha=git_commit_sha,
            )
        except (
            OSError,
            pd.errors.ParserError,
            SchemaValidationError,
            ValueError,
        ) as error:
            raise IntakeError(f"invalid House Prices dataset: {error}") from error

    payload = provenance.to_dict()
    payload.update(
        {
            "input_type": (
                "kaggle_zip" if input_file.suffix.lower() == ".zip" else "train_csv"
            ),
            "input_filename": "train.csv",
            "s3_object": {
                "bucket": s3_bucket,
                "key": s3_key or DEFAULT_S3_KEY_TEMPLATE.format(
                    dataset_version=dataset_version
                ),
            },
        }
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-url")
    parser.add_argument("--s3-bucket")
    parser.add_argument("--s3-key")
    parser.add_argument("--download-timestamp")
    parser.add_argument("--git-commit-sha")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = intake_dataset(
            args.input,
            dataset_version=args.dataset_version,
            output_dir=args.output_dir,
            source_url=args.source_url,
            s3_bucket=args.s3_bucket,
            s3_key=args.s3_key,
            download_timestamp=args.download_timestamp,
            git_commit_sha=args.git_commit_sha,
        )
    except IntakeError as error:
        print(f"dataset intake failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _validate_dataset_version(dataset_version: str) -> None:
    if not _VERSION_PATTERN.fullmatch(dataset_version):
        raise IntakeError(
            "dataset version must contain only letters, numbers, '.', '_' or '-'"
        )


def _validate_csv_header(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), None)
    if not header:
        raise SchemaValidationError("training CSV has no header")
    duplicates = [name for index, name in enumerate(header) if name in header[:index]]
    if duplicates:
        names = ", ".join(dict.fromkeys(duplicates))
        raise SchemaValidationError(f"duplicate columns: {names}")


def _source_url_for_input(input_path: Path) -> str:
    parts = {part.lower() for part in input_path.resolve().parts}
    if "fixtures" in parts and "tests" in parts:
        return FIXTURE_SOURCE_URL
    return DEFAULT_DATASET_SOURCE_URL


class _TrainCsvContext:
    def __init__(self, input_path: Path) -> None:
        self.input_path = input_path
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        if self.input_path.suffix.lower() != ".zip":
            if self.input_path.name != "train.csv":
                raise IntakeError("training input must be named train.csv")
            return self.input_path
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="house-prices-")
        destination = Path(self._temporary_directory.name) / "train.csv"
        try:
            with ZipFile(self.input_path) as archive:
                members = [
                    member
                    for member in archive.infolist()
                    if not member.is_dir() and Path(member.filename).name == "train.csv"
                ]
                if len(members) != 1:
                    raise IntakeError(
                        "Kaggle archive must contain exactly one train.csv"
                    )
                _reject_unsafe_archive_members(archive)
                destination.write_bytes(archive.read(members[0]))
        except BadZipFile as error:
            raise IntakeError("dataset input is not a valid ZIP archive") from error
        return destination

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()


def _materialize_train_csv(input_path: Path) -> _TrainCsvContext:
    return _TrainCsvContext(input_path)


def _reject_unsafe_archive_members(archive: ZipFile) -> None:
    for member in archive.infolist():
        candidate = Path(member.filename)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise IntakeError("dataset archive contains an unsafe path")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
