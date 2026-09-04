"""Tests for GaussianHMMDetector."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from skfolio_regime import GaussianHMMDetector
from skfolio_regime._gaussian_hmm import extract_regime_features, merge_short_runs


def _two_vol_regimes(n_low=80, n_high=80, n_assets=4, seed=0):
    rng = np.random.default_rng(seed)
    low = rng.normal(0.0, 0.005, size=(n_low, n_assets))
    high = rng.normal(0.0, 0.04, size=(n_high, n_assets))
    return np.vstack((low, high))


def test_extract_features_shapes():
    X = np.random.default_rng(0).normal(size=(30, 5))
    assert extract_regime_features(X, "mean").shape == (30, 1)
    assert extract_regime_features(X, "vol").shape == (30, 1)
    assert extract_regime_features(X, "mean_vol").shape == (30, 2)
    assert extract_regime_features(X, "full").shape == (30, 5)


def test_extract_features_invalid():
    X = np.zeros((10, 2))
    with pytest.raises(ValueError, match="feature"):
        extract_regime_features(X, "unknown")


def test_merge_short_runs_collapses_flicker():
    labels = np.array([0, 0, 0, 1, 0, 0, 0, 0])
    merged = merge_short_runs(labels, min_size=2)
    assert np.all(merged == 0)


def test_hmm_recovers_volatility_break():
    X = _two_vol_regimes()
    det = GaussianHMMDetector(
        n_regimes=2,
        feature="vol",
        min_regime_size=15,
        n_init=4,
        random_state=0,
    )
    det.fit(X)
    assert det.labels_.shape == (X.shape[0],)
    assert det.change_points_.size >= 1
    # Majority label should switch between the two halves.
    first = det.labels_[:80]
    second = det.labels_[80:]
    assert np.bincount(first).argmax() != np.bincount(second).argmax()


def test_hmm_reproducible_with_random_state():
    X = _two_vol_regimes()
    a = GaussianHMMDetector(n_regimes=2, feature="mean_vol", random_state=1).fit(X)
    b = GaussianHMMDetector(n_regimes=2, feature="mean_vol", random_state=1).fit(X)
    np.testing.assert_array_equal(a.labels_, b.labels_)
    np.testing.assert_allclose(a.log_likelihood_, b.log_likelihood_)


def test_hmm_clone_and_predict():
    X = _two_vol_regimes()
    det = GaussianHMMDetector(n_regimes=2, feature="vol", random_state=0)
    cloned = clone(det)
    cloned.fit(X)
    pred = cloned.predict(X)
    assert pred.shape == cloned.labels_.shape


def test_hmm_invalid_params():
    X = np.random.default_rng(0).normal(size=(20, 2))
    with pytest.raises(ValueError, match="n_regimes"):
        GaussianHMMDetector(n_regimes=0).fit(X)
    with pytest.raises(ValueError, match="covariance_type"):
        GaussianHMMDetector(covariance_type="banded").fit(X)


def test_hmm_single_regime():
    X = np.random.default_rng(0).normal(size=(40, 3))
    det = GaussianHMMDetector(n_regimes=1, min_regime_size=1, random_state=0)
    det.fit(X)
    assert np.all(det.labels_ == 0)
    assert det.change_points_.size == 0
