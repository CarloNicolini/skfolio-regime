"""Tests for GaussianHMMDetector."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

import skfolio_regime._gaussian_hmm as ghmm
from skfolio_regime import GaussianHMMDetector
from skfolio_regime._gaussian_hmm import (
    _EPS,
    _empirical_covariance,
    _forward_backward,
    _log_emissions,
    _rolling_std,
    extract_regime_features,
    fit_gaussian_hmm,
    merge_short_runs,
    standardize_features,
)


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


def test_rolling_std_is_causal():
    x = np.zeros(8)
    x[5:] = 10.0
    vol = _rolling_std(x, window=3)
    # Volatility at t < 5 must not see the jump (no bfill from later times).
    np.testing.assert_allclose(vol[:5], 0.0)
    assert vol[5] > 0.0
    # The last window is constant, so the causal std returns to 0.
    np.testing.assert_allclose(vol[-1], 0.0)


def test_extract_vol_feature_is_causal():
    X = np.zeros((8, 3))
    X[5:] = 0.1
    feat = extract_regime_features(X, "vol", vol_window=3).ravel()
    np.testing.assert_allclose(feat[:5], 0.0)
    assert feat[5] > 0.0


def test_standardize_features_reuses_training_stats():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(20, 2))
    b = a + 10.0
    za, mean, scale = standardize_features(a)
    zb, _, _ = standardize_features(b, mean=mean, scale=scale)
    z_re, _, _ = standardize_features(b)
    np.testing.assert_allclose(za.mean(axis=0), 0.0, atol=1e-12)
    assert np.linalg.norm(zb.mean(axis=0)) > 1.0
    np.testing.assert_allclose(z_re.mean(axis=0), 0.0, atol=1e-12)


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


def test_fit_hmm_likelihood_matches_returned_parameters():
    X = np.random.default_rng(12).normal(size=(80, 3))
    startprob, transmat, means, covars, reported = fit_gaussian_hmm(
        X,
        n_regimes=2,
        covariance_type="diag",
        n_iter=1,
        tol=1e-4,
        n_init=3,
        random_state=0,
    )
    emissions = _log_emissions(X, means, covars, "diag")
    actual, *_ = _forward_backward(
        np.log(np.maximum(startprob, _EPS)),
        np.log(np.maximum(transmat, _EPS)),
        emissions,
    )
    np.testing.assert_allclose(reported, actual)


def test_fit_hmm_ignores_failed_initialization(monkeypatch):
    X = np.random.default_rng(13).normal(size=(60, 2))
    original = ghmm._init_hmm
    calls = 0

    def fail_first_initialization(*args, **kwargs):
        nonlocal calls
        calls += 1
        startprob, transmat, means, covars = original(*args, **kwargs)
        if calls == 1:
            means[:] = np.nan
        return startprob, transmat, means, covars

    monkeypatch.setattr(ghmm, "_init_hmm", fail_first_initialization)
    *_, log_likelihood = fit_gaussian_hmm(
        X,
        n_regimes=2,
        covariance_type="diag",
        n_iter=3,
        tol=1e-4,
        n_init=2,
        random_state=0,
    )
    assert np.isfinite(log_likelihood)


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


def test_extract_features_invalid():
    X = np.zeros((10, 2))
    with pytest.raises(ValueError, match="feature"):
        extract_regime_features(X, "unknown")


def test_hmm_predict_in_sample_matches_labels():
    X = _two_vol_regimes()
    det = GaussianHMMDetector(
        n_regimes=2, feature="vol", min_regime_size=10, random_state=0
    ).fit(X)
    np.testing.assert_array_equal(det.predict(X), det.labels_)


def test_hmm_predict_uses_training_scaler():
    rng = np.random.default_rng(1)
    X = rng.normal(0.0, 0.01, size=(80, 3))
    det = GaussianHMMDetector(
        n_regimes=2,
        feature="mean",
        min_regime_size=1,
        n_init=3,
        random_state=0,
    ).fit(X)
    np.testing.assert_array_equal(det.predict(X), det.labels_)
    shifted = X + 5.0
    raw_shift = extract_regime_features(shifted, "mean")
    frozen, _, _ = standardize_features(
        raw_shift, mean=det.feature_mean_, scale=det.feature_scale_
    )
    recomputed, _, _ = standardize_features(raw_shift)
    # Frozen training scale keeps the location shift; re-standardizing would
    # erase it and leak evaluation-set moments into decoding.
    assert not np.allclose(frozen, recomputed)


def test_hmm_single_observation():
    X = np.array([[0.01, -0.02, 0.0]])
    det = GaussianHMMDetector(n_regimes=2, min_regime_size=21, n_init=2, random_state=0)
    det.fit(X)
    assert det.labels_.shape == (1,)
    assert np.isfinite(det.startprob_).all()
    assert np.isfinite(det.transmat_).all()
    assert det.predict(X).shape == (1,)


def test_hmm_constant_returns():
    X = np.zeros((40, 4))
    det = GaussianHMMDetector(n_regimes=2, feature="mean_vol", n_init=2, random_state=0)
    det.fit(X)
    assert det.labels_.shape == (40,)
    assert np.isfinite(det.log_likelihood_) or det.log_likelihood_ == -np.inf


def test_hmm_full_covariance_one_feature():
    X = _two_vol_regimes(n_low=40, n_high=40, n_assets=3, seed=3)
    cov = _empirical_covariance(extract_regime_features(X, "vol"))
    assert cov.shape == (1, 1)
    det = GaussianHMMDetector(
        n_regimes=2,
        feature="vol",
        covariance_type="full",
        min_regime_size=8,
        n_init=3,
        random_state=0,
    )
    det.fit(X)
    assert det.covars_.shape[-2:] == (1, 1)
    assert np.isfinite(det.covars_).all()


def test_hmm_spherical_and_diag_fit():
    X = _two_vol_regimes(n_low=50, n_high=50, seed=4)
    for cov in ("diag", "spherical"):
        det = GaussianHMMDetector(
            n_regimes=2,
            feature="mean_vol",
            covariance_type=cov,
            min_regime_size=10,
            n_init=2,
            random_state=0,
        )
        det.fit(X)
        assert det.labels_.shape[0] == X.shape[0]
        assert np.isfinite(det.transmat_).all()


def test_hmm_nan_and_inf_returns():
    X = _two_vol_regimes(n_low=30, n_high=30, seed=5)
    X[3, 0] = np.nan
    X[8, 1] = np.inf
    det = GaussianHMMDetector(
        n_regimes=2, feature="mean", min_regime_size=5, n_init=2, random_state=0
    )
    det.fit(X)
    assert det.labels_.shape[0] == X.shape[0]


def test_hmm_dataframe_input():
    import pandas as pd

    X = pd.DataFrame(_two_vol_regimes(n_low=40, n_high=40, n_assets=3, seed=6))
    det = GaussianHMMDetector(
        n_regimes=2, feature="vol", min_regime_size=8, n_init=2, random_state=0
    )
    det.fit(X)
    assert det.n_features_in_ == 3
    np.testing.assert_array_equal(det.predict(X), det.labels_)


def test_hmm_transmat_is_stochastic():
    X = _two_vol_regimes()
    det = GaussianHMMDetector(n_regimes=2, feature="vol", random_state=0).fit(X)
    np.testing.assert_allclose(det.transmat_.sum(axis=1), 1.0, atol=1e-8)
    np.testing.assert_allclose(det.startprob_.sum(), 1.0, atol=1e-8)
