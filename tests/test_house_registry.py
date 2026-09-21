"""Tests for explicit House Pricing model registration and promotion."""

import json
from pathlib import Path

import pytest

from house_pricing_mlops.registry import (
    RegistryError,
    VersionedModelRegistry,
)


def _candidate(registry: VersionedModelRegistry) -> dict[str, object]:
    return registry.register_candidate(
        model_name="house-price-model",
        model_version="v1",
        run_id="run-123",
        dataset_version="v1",
        dataset_sha256="dataset-sha",
        bundle_sha256="bundle-sha",
        s3_bucket="private-bucket",
        s3_key="models/house-price/v1/model.tar.gz",
        s3_version_id="object-version-1",
        selected_model="gradient_boosting",
        metrics={"mae": 1000.0, "rmse": 1500.0, "r2": 0.9, "rmsle": 0.1},
    )


def test_candidate_registration_retains_immutable_artifact_coordinates(
    tmp_path: Path,
) -> None:
    registry = VersionedModelRegistry(tmp_path / "registry.json")

    record = _candidate(registry)
    state = registry.read()

    assert record["registry_version"] == "1"
    assert record["status"] == "candidate"
    assert state["aliases"] == {"candidate": "1"}
    assert record["s3_object"] == {
        "bucket": "private-bucket",
        "key": "models/house-price/v1/model.tar.gz",
        "version_id": "object-version-1",
    }
    assert state.get("aliases", {}).get("champion") is None


def test_promotion_requires_explicit_approval_and_does_not_change_candidate_identity(
    tmp_path: Path,
) -> None:
    registry = VersionedModelRegistry(tmp_path / "registry.json")
    record = _candidate(registry)

    with pytest.raises(RegistryError, match="approved_by"):
        registry.promote_champion(
            record["registry_version"], approved_by="", reason="approved"
        )

    promoted = registry.promote_champion(
        record["registry_version"],
        approved_by="operator@example.com",
        reason="reviewed",
    )
    state = registry.read()

    assert promoted["status"] == "champion"
    assert state["aliases"] == {"candidate": "1", "champion": "1"}
    assert state["promotion_decisions"][-1]["reason"] == "reviewed"
    assert state["promotion_decisions"][-1]["previous_champion"] is None


def test_registry_rejects_mutable_latest_and_records_rollback_metadata(
    tmp_path: Path,
) -> None:
    registry = VersionedModelRegistry(tmp_path / "registry.json")
    first = _candidate(registry)
    registry.promote_champion(
        first["registry_version"], approved_by="a", reason="first"
    )
    second = registry.register_candidate(
        model_name="house-price-model",
        model_version="v2",
        run_id="run-456",
        dataset_version="v2",
        dataset_sha256="dataset-sha-2",
        bundle_sha256="bundle-sha-2",
        s3_bucket="private-bucket",
        s3_key="models/house-price/v2/model.tar.gz",
        s3_version_id="object-version-2",
        selected_model="random_forest",
        metrics={},
    )
    registry.promote_champion(
        second["registry_version"], approved_by="b", reason="second"
    )

    with pytest.raises(RegistryError, match="exact S3 VersionId"):
        registry.register_candidate(
            model_name="house-price-model",
            model_version="v3",
            run_id="run-789",
            dataset_version="v3",
            dataset_sha256="dataset-sha-3",
            bundle_sha256="bundle-sha-3",
            s3_bucket="private-bucket",
            s3_key="models/house-price/latest/model.tar.gz",
            s3_version_id="object-version-3",
            selected_model="dummy",
            metrics={},
        )

    rolled_back = registry.rollback_champion(
        first["registry_version"], approved_by="rollback@example.com", reason="revert"
    )
    state = registry.read()

    assert rolled_back["registry_version"] == first["registry_version"]
    assert state["aliases"]["champion"] == first["registry_version"]
    assert state["promotion_decisions"][-1]["action"] == "rollback"
    assert (
        state["promotion_decisions"][-1]["previous_champion"]
        == second["registry_version"]
    )
    assert json.loads((tmp_path / "registry.json").read_text()) == state
