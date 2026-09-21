"""Tests for deterministic House Pricing training."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from house_pricing_mlops.training import (
    CANDIDATE_MODEL_NAMES,
    TrainingConfig,
    train_house_price_model,
)

FIXTURE_PATH = Path("tests/fixtures/house_prices/train.csv")


def _fixture_frame() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_PATH)


def test_split_is_reproducible_for_a_fixed_seed() -> None:
    config = TrainingConfig(random_seed=19)

    first = train_house_price_model(_fixture_frame(), config=config)
    second = train_house_price_model(_fixture_frame(), config=config)

    assert first.split == second.split
    assert first.selected_model_name == second.selected_model_name
    np.testing.assert_allclose(
        first.selected_pipeline.predict(first.selected_features),
        second.selected_pipeline.predict(second.selected_features),
    )


def test_split_changes_when_seed_changes() -> None:
    first = train_house_price_model(
        _fixture_frame(), config=TrainingConfig(random_seed=19)
    )
    second = train_house_price_model(
        _fixture_frame(), config=TrainingConfig(random_seed=20)
    )

    assert first.split != second.split


def test_training_compares_all_declared_candidates() -> None:
    result = train_house_price_model(
        _fixture_frame(), config=TrainingConfig(random_seed=19)
    )

    assert tuple(candidate.name for candidate in result.candidates) == (
        *CANDIDATE_MODEL_NAMES,
    )
    assert result.selected_model_name in CANDIDATE_MODEL_NAMES
    assert result.selected_candidate.validation_metrics["rmse"] == min(
        candidate.validation_metrics["rmse"] for candidate in result.candidates
    )


def test_preprocessing_does_not_leak_test_values() -> None:
    frame = _fixture_frame()
    baseline = train_house_price_model(frame, config=TrainingConfig(random_seed=19))
    mutated = frame.copy()
    mutated.loc[list(baseline.split.test_indices), "GrLivArea"] = 999999.0
    changed_test = train_house_price_model(
        mutated, config=TrainingConfig(random_seed=19)
    )

    baseline_dummy = next(
        candidate for candidate in baseline.candidates if candidate.name == "dummy"
    )
    changed_dummy = next(
        candidate for candidate in changed_test.candidates if candidate.name == "dummy"
    )
    baseline_preprocessor = baseline_dummy.pipeline.regressor_.named_steps[
        "preprocessor"
    ]
    changed_preprocessor = changed_dummy.pipeline.regressor_.named_steps[
        "preprocessor"
    ]
    baseline_imputer = baseline_preprocessor.transformers_[0][1].named_steps[
        "imputer"
    ]
    changed_imputer = changed_preprocessor.transformers_[0][1].named_steps[
        "imputer"
    ]

    np.testing.assert_allclose(
        baseline_imputer.statistics_, changed_imputer.statistics_
    )


def test_unknown_category_is_handled_by_fitted_pipeline() -> None:
    result = train_house_price_model(
        _fixture_frame(), config=TrainingConfig(random_seed=19)
    )
    row = result.selected_features.iloc[:1].copy()
    row.loc[:, "Neighborhood"] = "UnknownNeighborhood"

    prediction = result.selected_pipeline.predict(row)

    assert len(prediction) == 1
    assert np.isfinite(prediction[0])


@pytest.mark.parametrize("invalid_seed", [True, -1])
def test_training_config_rejects_invalid_seed(invalid_seed: object) -> None:
    with pytest.raises(ValueError, match="random_seed"):
        TrainingConfig(random_seed=invalid_seed)  # type: ignore[arg-type]
