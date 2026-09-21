"""Dataset provenance and checksum capture for House Pricing intake."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from house_pricing_mlops.schema import (
    TARGET_NAME,
    feature_schema_dict,
    validate_raw_frame,
)

DEFAULT_DATASET_SOURCE_URL = (
    "https://www.kaggle.com/competitions/"
    "house-prices-advanced-regression-techniques/data"
)


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    """Immutable metadata describing one dataset input."""

    source_url: str
    dataset_version: str
    download_timestamp: str
    dataset_sha256: str
    row_count: int
    column_count: int
    feature_schema: Mapping[str, object]
    target_column: str
    git_commit_sha: str
    training_configuration: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe provenance mapping."""

        return {
            "source_url": self.source_url,
            "dataset_version": self.dataset_version,
            "download_timestamp": self.download_timestamp,
            "dataset_sha256": self.dataset_sha256,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "feature_schema": json.loads(json.dumps(self.feature_schema)),
            "target_column": self.target_column,
            "git_commit_sha": self.git_commit_sha,
            "training_configuration": json.loads(
                json.dumps(self.training_configuration)
            ),
        }


def sha256_file(path: str | Path) -> str:
    """Calculate a streaming SHA-256 digest for one local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_provenance(
    path: str | Path,
    *,
    dataset_version: str,
    source_url: str = DEFAULT_DATASET_SOURCE_URL,
    download_timestamp: str | None = None,
    git_commit_sha: str | None = None,
    training_configuration: Mapping[str, object] | None = None,
) -> DatasetProvenance:
    """Validate a CSV and capture all required provenance fields."""

    dataset_path = Path(path)
    frame = pd.read_csv(dataset_path)
    validated = validate_raw_frame(frame)
    timestamp = download_timestamp or datetime.now(UTC).isoformat()
    commit_sha = git_commit_sha or os.environ.get("GIT_COMMIT_SHA") or _git_sha()
    configuration = dict(training_configuration or {})
    return DatasetProvenance(
        source_url=source_url,
        dataset_version=dataset_version,
        download_timestamp=timestamp,
        dataset_sha256=sha256_file(dataset_path),
        row_count=validated.row_count,
        column_count=validated.raw_column_count,
        feature_schema=feature_schema_dict(),
        target_column=TARGET_NAME,
        git_commit_sha=commit_sha,
        training_configuration=configuration,
    )


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


__all__ = [
    "DEFAULT_DATASET_SOURCE_URL",
    "DatasetProvenance",
    "build_dataset_provenance",
    "sha256_file",
]
