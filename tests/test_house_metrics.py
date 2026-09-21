"""Tests for House Pricing evaluation metrics."""

import numpy as np
import pytest

from house_pricing_mlops.training import calculate_metrics


def test_calculate_metrics_returns_original_unit_metrics() -> None:
    metrics = calculate_metrics(
        np.array([100.0, 200.0, 300.0]),
        np.array([110.0, 190.0, 300.0]),
    )

    assert metrics["mae"] == pytest.approx(6.6666667)
    assert metrics["rmse"] == pytest.approx(8.1649658)
    assert metrics["r2"] == pytest.approx(0.99)
    assert metrics["rmsle"] is not None


def test_calculate_metrics_omits_rmsle_for_negative_predictions() -> None:
    metrics = calculate_metrics(np.array([100.0, 200.0]), np.array([-1.0, 200.0]))

    assert metrics["rmsle"] is None
