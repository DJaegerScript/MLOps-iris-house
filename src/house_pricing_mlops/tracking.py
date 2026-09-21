"""Small MLflow boundary for House Pricing experiments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from house_pricing_mlops.provenance import DatasetProvenance
    from house_pricing_mlops.training import TrainingResult


@dataclass(frozen=True, slots=True)
class TrackingResult:
    """Safe identifiers returned after an MLflow run completes."""

    run_id: str
    experiment_id: str
    tracking_uri: str
    registered_model_version: str | None = None


class MLflowTracker:
    """Log one training result to local or configured MLflow."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        *,
        experiment_name: str = "house-prices",
    ) -> None:
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "file:./mlruns"
        )
        self.experiment_name = experiment_name

    def __repr__(self) -> str:
        return (
            f"MLflowTracker(experiment_name={self.experiment_name!r}, "
            f"tracking_uri={_redact_uri(self.tracking_uri)!r})"
        )

    def log(
        self,
        result: TrainingResult,
        provenance: DatasetProvenance,
        *,
        training_duration_seconds: float,
        bundle_path: str | Path | None = None,
        registered_model_name: str | None = None,
    ) -> TrackingResult:
        """Log required parameters, tags, metrics, and an optional bundle."""

        import mlflow

        mlflow.set_tracking_uri(self.tracking_uri)
        experiment = mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run() as run:
            params: dict[str, object] = {
                "dataset_version": provenance.dataset_version,
                "random_seed": result.config.random_seed,
                "selected_model": result.selected_model_name,
                "model_schema_version": provenance.feature_schema["version"],
                "target_transform": "log1p_expm1",
                "validation_size": result.config.validation_size,
                "test_size": result.config.test_size,
            }
            for candidate in result.candidates:
                for name, value in candidate.parameters.items():
                    params[f"{candidate.name}_param_{name}"] = value
            mlflow.log_params(params)
            mlflow.set_tags(
                {
                    "dataset_sha256": provenance.dataset_sha256,
                    "git_commit_sha": provenance.git_commit_sha,
                    "feature_list": json.dumps(
                        provenance.feature_schema, sort_keys=True
                    ),
                    "preprocessing_configuration": json.dumps(
                        {
                            "numeric_imputer": "median",
                            "categorical_imputer": "__MISSING__",
                            "unknown_category": "ignore",
                        },
                        sort_keys=True,
                    ),
                    "target_name": provenance.target_column,
                    "source_url": provenance.source_url,
                }
            )
            metrics: dict[str, float] = {
                "training_duration_seconds": float(training_duration_seconds)
            }
            for metric_name, value in result.test_metrics.items():
                if value is not None:
                    metrics[f"test_{metric_name}"] = float(value)
            for candidate in result.candidates:
                for metric_name, value in candidate.validation_metrics.items():
                    if value is not None:
                        metrics[
                            f"{candidate.name}_validation_{metric_name}"
                        ] = float(value)
            mlflow.log_metrics(metrics)
            registered_model_version: str | None = None
            if bundle_path is not None:
                mlflow.log_artifact(str(bundle_path), artifact_path="model-bundle")
            if registered_model_name is not None:
                model_info = mlflow.sklearn.log_model(
                    result.selected_pipeline,
                    artifact_path="selected-model",
                    registered_model_name=registered_model_name,
                )
                registered_model_version = getattr(
                    model_info, "registered_model_version", None
                )
            return TrackingResult(
                run_id=run.info.run_id,
                experiment_id=experiment.experiment_id,
                tracking_uri=self.tracking_uri,
                registered_model_version=(
                    str(registered_model_version)
                    if registered_model_version is not None
                    else None
                ),
            )


def _redact_uri(uri: str) -> str:
    if "@" not in uri or "://" not in uri:
        return uri
    scheme, remainder = uri.split("://", 1)
    _, host = remainder.rsplit("@", 1)
    return f"{scheme}://***@{host}"


__all__ = ["MLflowTracker", "TrackingResult"]
