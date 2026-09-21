"""Tests for the offline, checksum-pinned model intake packaging step."""

from pathlib import Path

import pytest

from scripts.package_model_bundle import (
    IntakeValidationError,
    validate_intake_directory,
)

ARTIFACT_NAMES = (
    "iris_model.pkl",
    "scaler.pkl",
    "label_encoder.pkl",
    "metadata.pkl",
)


def test_intake_rejects_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(IntakeValidationError, match="missing intake artifact"):
        validate_intake_directory(tmp_path)


def test_intake_rejects_unexpected_file(tmp_path: Path) -> None:
    for name in ARTIFACT_NAMES:
        (tmp_path / name).write_bytes(b"not a verified artifact")
    (tmp_path / "dataset.csv").write_text("sepal_length\n", encoding="utf-8")

    with pytest.raises(IntakeValidationError, match="unexpected intake files"):
        validate_intake_directory(tmp_path)


def test_intake_rejects_checksum_mismatch(tmp_path: Path) -> None:
    for name in ARTIFACT_NAMES:
        (tmp_path / name).write_bytes(b"not a verified artifact")

    with pytest.raises(IntakeValidationError, match="checksum mismatch"):
        validate_intake_directory(tmp_path)
