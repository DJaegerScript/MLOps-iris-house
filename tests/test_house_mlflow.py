"""Tests for local MLflow experiment tracking."""

from pathlib import Path

import mlflow
import pandas as pd

from house_pricing_mlops.provenance import build_dataset_provenance
from house_pricing_mlops.tracking import MLflowTracker
from house_pricing_mlops.training import TrainingConfig, train_house_price_model

FIXTURE_PATH = Path("tests/fixtures/house_prices/train.csv")


def test_tracker_logs_required_parameters_tags_and_metrics(tmp_path: Path) -> None:
    tracking_uri = (tmp_path / "mlruns").as_uri()
    result = train_house_price_model(
        pd.read_csv(FIXTURE_PATH), config=TrainingConfig(random_seed=7)
    )
    provenance = build_dataset_provenance(
        FIXTURE_PATH,
        dataset_version="fixture-v1",
        download_timestamp="2026-09-21T00:00:00+00:00",
        git_commit_sha="abc123",
        training_configuration=result.config.to_dict(),
    )

    tracked = MLflowTracker(tracking_uri, experiment_name="house-prices-test").log(
        result,
        provenance,
        training_duration_seconds=1.25,
    )

    run = mlflow.MlflowClient(tracking_uri=tracking_uri).get_run(tracked.run_id)
    assert run.data.params["dataset_version"] == "fixture-v1"
    assert run.data.params["random_seed"] == "7"
    assert run.data.params["selected_model"] == result.selected_model_name
    assert run.data.params["random_forest_param_n_estimators"] == "40"
    assert run.data.tags["dataset_sha256"] == provenance.dataset_sha256
    assert run.data.tags["git_commit_sha"] == "abc123"
    assert "feature_list" in run.data.tags
    assert run.data.metrics["test_mae"] == result.test_metrics["mae"]
    assert run.data.metrics["training_duration_seconds"] == 1.25


def test_tracker_exposes_configured_remote_uri_without_logging_credentials() -> None:
    tracker = MLflowTracker(
        "https://mlflow.example.invalid/tracking",
        experiment_name="house-prices",
    )

    assert tracker.tracking_uri == "https://mlflow.example.invalid/tracking"
    assert "password" not in repr(tracker).lower()


def test_tracker_registers_selected_model_when_requested(tmp_path: Path) -> None:
    tracking_uri = (tmp_path / "mlruns").as_uri()
    result = train_house_price_model(
        pd.read_csv(FIXTURE_PATH), config=TrainingConfig(random_seed=7)
    )
    provenance = build_dataset_provenance(
        FIXTURE_PATH,
        dataset_version="fixture-v1",
        download_timestamp="2026-09-21T00:00:00+00:00",
        git_commit_sha="abc123",
        training_configuration=result.config.to_dict(),
    )

    tracked = MLflowTracker(
        tracking_uri,
        experiment_name="house-prices-registry-test",
    ).log(
        result,
        provenance,
        training_duration_seconds=1.25,
        registered_model_name="house-price-model-test",
    )

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions("name = 'house-price-model-test'")
    assert len(versions) == 1
    assert versions[0].run_id == tracked.run_id
    assert versions[0].name == "house-price-model-test"
