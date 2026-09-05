"""Tests for the synthetic stress-test harness."""

from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

BENCHMARK = run_path(
    Path(__file__).parents[1] / "benchmarks" / "regime_stress.py"
)


def test_synthetic_scenario_contracts():
    BENCHMARK["validate_scenarios"]()


def test_stationary_null_has_no_economic_regime_labels():
    path = BENCHMARK["generate_path"](
        "stationary_null",
        path_seed=0,
        n_observations=200,
    )
    assert np.all(path.states == 0)
    assert np.isfinite(path.returns).all()


def test_current_regime_is_a_suffix():
    labels = np.array([0, 0, 1, 1, 0, 0])
    positions, fallback = BENCHMARK["_filter_positions"](
        labels,
        scope="current_regime",
        min_train_size=1,
    )
    np.testing.assert_array_equal(positions, [4, 5])
    assert not fallback


def test_portfolio_metrics_reward_constant_positive_returns():
    metrics = BENCHMARK["_portfolio_metrics"](np.full(100, 0.001))
    assert metrics["annual_return"] == pytest.approx(0.252)
    assert metrics["max_drawdown"] == 0.0
    assert np.isnan(metrics["sharpe"])
