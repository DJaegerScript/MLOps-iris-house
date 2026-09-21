"""Tests for House Pricing bundle creation and integrity metadata."""

import io
import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from house_pricing_mlops.bundle import (
    BUNDLE_MEMBERS,
    BundleIntegrityError,
    create_model_bundle,
    load_model_bundle,
)
from house_pricing_mlops.provenance import build_dataset_provenance
from house_pricing_mlops.training import TrainingConfig, train_house_price_model

FIXTURE_PATH = Path("tests/fixtures/house_prices/train.csv")


def _trained(tmp_path: Path):
    frame = pd.read_csv(FIXTURE_PATH)
    result = train_house_price_model(frame, config=TrainingConfig(random_seed=7))
    provenance = build_dataset_provenance(
        FIXTURE_PATH,
        dataset_version="fixture-v1",
        download_timestamp="2026-09-21T00:00:00+00:00",
        git_commit_sha="abc123",
        training_configuration=result.config.to_dict(),
    )
    artifact = create_model_bundle(
        result,
        provenance,
        output_path=tmp_path / "model.tar.gz",
        model_version="v1",
    )
    return frame, result, artifact


def test_bundle_contains_exact_members_and_manifest_metadata(tmp_path: Path) -> None:
    _frame, result, artifact = _trained(tmp_path)

    with tarfile.open(artifact.path, mode="r:gz") as archive:
        assert tuple(sorted(member.name for member in archive.getmembers())) == tuple(
            sorted(BUNDLE_MEMBERS)
        )
        manifest = json.load(archive.extractfile("manifest.json"))

    assert manifest["model_name"] == "house-price-model"
    assert manifest["model_version"] == "v1"
    assert manifest["dataset_version"] == "fixture-v1"
    assert manifest["git_commit_sha"] == "abc123"
    assert manifest["feature_names"]
    assert manifest["target_name"] == "SalePrice"
    assert manifest["model_type"] == result.selected_model_name
    assert manifest["bundle_sha256"] == artifact.bundle_sha256
    assert set(manifest["artifact_checksums"]) == {
        name for name in BUNDLE_MEMBERS if name != "manifest.json"
    }


def test_valid_bundle_loads_complete_pipeline_and_predicts(tmp_path: Path) -> None:
    frame, _result, artifact = _trained(tmp_path)

    loaded = load_model_bundle(artifact.path)
    prediction = loaded.predict(frame.loc[:0, loaded.feature_schema["feature_names"]])

    assert len(prediction) == 1
    assert prediction[0] > 0
    assert loaded.manifest["model_version"] == "v1"


def test_tampered_member_is_rejected_before_deserialization(tmp_path: Path) -> None:
    _frame, _result, artifact = _trained(tmp_path)
    tampered = tmp_path / "tampered.tar.gz"
    with tarfile.open(artifact.path, mode="r:gz") as source, tarfile.open(
        tampered, mode="w:gz"
    ) as destination:
        for member in source.getmembers():
            payload = source.extractfile(member).read()
            if member.name == "model.joblib":
                payload += b"tampered"
                member.size = len(payload)
            destination.addfile(member, io.BytesIO(payload))

    with pytest.raises(BundleIntegrityError, match="checksum"):
        load_model_bundle(tampered)


def test_archive_path_traversal_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(path, mode="w:gz") as archive:
        payload = b"unsafe"
        member = tarfile.TarInfo("../model.joblib")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(BundleIntegrityError, match="path"):
        load_model_bundle(path)
