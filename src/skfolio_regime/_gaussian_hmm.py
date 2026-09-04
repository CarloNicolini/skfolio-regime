"""Gaussian hidden Markov model regime detector."""

# Copyright (c) 2026
# Author: Carlo Nicolini <c.nicolini@ipazia.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import sklearn.cluster as skc
import sklearn.utils as sku
import sklearn.utils.validation as skv
from scipy.special import logsumexp
from skfolio.typing import ArrayLike, FloatArray, IntArray

from skfolio_regime._base import BaseRegimeDetector

_MIN_COVAR = 1e-6
_EPS = 1e-12


def _rolling_std(x: FloatArray, window: int) -> FloatArray:
    """Causal rolling standard deviation, including the current observation."""
    series = pd.Series(np.asarray(x, dtype=float))
    vol = series.rolling(window=window, min_periods=2).std(ddof=0)
    return vol.bfill().fillna(0.0).to_numpy(dtype=float)


def extract_regime_features(
    X: FloatArray,
    feature: str,
    vol_window: int = 21,
) -> FloatArray:
    """Build a low-dimensional feature matrix from asset returns.

    Parameters
    ----------
    X : ndarray of shape (n_observations, n_assets)
        Asset returns.

    feature : {"mean", "vol", "mean_vol", "full"}
        Feature construction:

        * ``mean``: cross-sectional mean return.
        * ``vol``: rolling standard deviation of the cross-sectional mean.
        * ``mean_vol``: both of the above.
        * ``full``: the raw return matrix.

    vol_window : int, default=21
        Window length used for the realized-volatility feature.

    Returns
    -------
    features : ndarray of shape (n_observations, n_features)
        Standardized features (zero mean, unit variance per column).
    """
    x = np.asarray(X, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"X must be 2d, got shape {x.shape}")
    cs_mean = np.nanmean(x, axis=1)
    if feature == "mean":
        features = cs_mean[:, np.newaxis]
    elif feature == "vol":
        features = _rolling_std(cs_mean, vol_window)[:, np.newaxis]
    elif feature == "mean_vol":
        features = np.column_stack((cs_mean, _rolling_std(cs_mean, vol_window)))
    elif feature == "full":
        features = np.nan_to_num(x, nan=0.0)
    else:
        raise ValueError(
            f"`feature` must be 'mean', 'vol', 'mean_vol' or 'full', got {feature!r}."
        )
    features = np.nan_to_num(features, nan=0.0)
    scale = features.std(axis=0, keepdims=True)
    scale = np.where(scale < _EPS, 1.0, scale)
    features = (features - features.mean(axis=0, keepdims=True)) / scale
    return features


def merge_short_runs(labels: IntArray, min_size: int) -> IntArray:
    """Merge consecutive runs shorter than ``min_size`` into a neighbor."""
    labels = np.asarray(labels, dtype=int).copy()
    if min_size <= 1 or labels.size == 0:
        return labels

    changed = True
    while changed:
        changed = False
        runs: list[tuple[int, int, int]] = []
        start = 0
        for i in range(1, labels.size + 1):
            if i == labels.size or labels[i] != labels[start]:
                runs.append((start, i, int(labels[start])))
                start = i
        if len(runs) <= 1:
            break
        for i, (start, end, _lab) in enumerate(runs):
            if end - start >= min_size:
                continue
            left_len = runs[i - 1][1] - runs[i - 1][0] if i > 0 else -1
            right_len = runs[i + 1][1] - runs[i + 1][0] if i < len(runs) - 1 else -1
            if right_len >= left_len and i < len(runs) - 1:
                labels[start:end] = runs[i + 1][2]
            elif i > 0:
                labels[start:end] = runs[i - 1][2]
            else:
                continue
            changed = True
            break
    return labels


def _log_diag_gaussian(
    X: FloatArray, means: FloatArray, covars: FloatArray
) -> FloatArray:
    """Log density of diagonal Gaussians. Returns shape (n_obs, n_regimes)."""
    n_regimes, n_features = means.shape
    log_prob = np.empty((X.shape[0], n_regimes))
    for k in range(n_regimes):
        var = np.maximum(covars[k], _MIN_COVAR)
        diff = X - means[k]
        log_prob[:, k] = -0.5 * (
            n_features * np.log(2.0 * np.pi)
            + np.sum(np.log(var))
            + np.sum(diff**2 / var, axis=1)
        )
    return log_prob


def _log_full_gaussian(
    X: FloatArray, means: FloatArray, covars: FloatArray
) -> FloatArray:
    """Log density of full-covariance Gaussians."""
    n_regimes = means.shape[0]
    n_features = X.shape[1]
    log_prob = np.empty((X.shape[0], n_regimes))
    for k in range(n_regimes):
        cov = covars[k] + _MIN_COVAR * np.eye(n_features)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            logdet = n_features * np.log(_MIN_COVAR)
            cov = _MIN_COVAR * np.eye(n_features)
        diff = X - means[k]
        try:
            sol = np.linalg.solve(cov, diff.T).T
        except np.linalg.LinAlgError:
            sol = diff / _MIN_COVAR
        log_prob[:, k] = -0.5 * (
            n_features * np.log(2.0 * np.pi) + logdet + np.sum(diff * sol, axis=1)
        )
    return log_prob


def _log_emissions(
    X: FloatArray,
    means: FloatArray,
    covars: FloatArray,
    covariance_type: str,
) -> FloatArray:
    if covariance_type == "diag":
        return _log_diag_gaussian(X, means, covars)
    if covariance_type == "full":
        return _log_full_gaussian(X, means, covars)
    if covariance_type == "spherical":
        n_features = X.shape[1]
        covars_diag = np.repeat(np.asarray(covars)[:, np.newaxis], n_features, axis=1)
        return _log_diag_gaussian(X, means, covars_diag)
    raise ValueError(f"Unknown covariance_type {covariance_type!r}.")


def _forward_backward(
    log_start: FloatArray,
    log_trans: FloatArray,
    log_emit: FloatArray,
) -> tuple[float, FloatArray, FloatArray]:
    """Forward-backward in log space.

    Returns
    -------
    log_likelihood : float
    log_gamma : ndarray of shape (n_obs, n_regimes)
    log_xi : ndarray of shape (n_obs - 1, n_regimes, n_regimes)
    """
    n_obs, n_regimes = log_emit.shape
    log_alpha = np.empty((n_obs, n_regimes))
    log_alpha[0] = log_start + log_emit[0]
    for t in range(1, n_obs):
        log_alpha[t] = log_emit[t] + logsumexp(
            log_alpha[t - 1][:, np.newaxis] + log_trans, axis=0
        )

    log_likelihood = float(logsumexp(log_alpha[-1]))

    log_beta = np.empty((n_obs, n_regimes))
    log_beta[-1] = 0.0
    for t in range(n_obs - 2, -1, -1):
        log_beta[t] = logsumexp(log_trans + log_emit[t + 1] + log_beta[t + 1], axis=1)

    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)

    log_xi = (
        log_alpha[:-1, :, np.newaxis]
        + log_trans[np.newaxis, :, :]
        + log_emit[1:, np.newaxis, :]
        + log_beta[1:, np.newaxis, :]
    )
    log_xi -= logsumexp(log_xi, axis=(1, 2), keepdims=True)
    return log_likelihood, log_gamma, log_xi


def _viterbi(
    log_start: FloatArray,
    log_trans: FloatArray,
    log_emit: FloatArray,
) -> IntArray:
    n_obs, n_regimes = log_emit.shape
    log_delta = np.empty((n_obs, n_regimes))
    psi = np.zeros((n_obs, n_regimes), dtype=int)
    log_delta[0] = log_start + log_emit[0]
    for t in range(1, n_obs):
        scores = log_delta[t - 1][:, np.newaxis] + log_trans
        psi[t] = np.argmax(scores, axis=0)
        log_delta[t] = log_emit[t] + np.max(scores, axis=0)
    states = np.empty(n_obs, dtype=int)
    states[-1] = int(np.argmax(log_delta[-1]))
    for t in range(n_obs - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


def _init_hmm(
    X: FloatArray,
    n_regimes: int,
    covariance_type: str,
    random_state,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    n_obs, n_features = X.shape
    rng = sku.check_random_state(random_state)
    means = np.zeros((n_regimes, n_features))
    if n_obs < n_regimes:
        means[:n_obs] = X
        means[n_obs:] = X[-1]
    else:
        try:
            kmeans = skc.KMeans(
                n_clusters=n_regimes,
                n_init=1,
                random_state=rng.randint(0, np.iinfo(np.int32).max),
            )
            labels = kmeans.fit_predict(X)
            for k in range(n_regimes):
                mask = labels == k
                if np.any(mask):
                    means[k] = X[mask].mean(axis=0)
                else:
                    means[k] = X[rng.randint(0, n_obs)]
        except ValueError:
            means = X[rng.choice(n_obs, size=n_regimes, replace=True)]

    transmat = np.full((n_regimes, n_regimes), 0.1 / max(n_regimes - 1, 1))
    np.fill_diagonal(transmat, 0.9 if n_regimes > 1 else 1.0)
    startprob = np.full(n_regimes, 1.0 / n_regimes)

    if covariance_type == "full":
        covars = np.array([np.cov(X.T) + _MIN_COVAR * np.eye(n_features)] * n_regimes)
        if covars.ndim == 2:
            covars = np.stack([covars] * n_regimes)
    elif covariance_type == "spherical":
        covars = np.full(n_regimes, max(float(np.var(X)), _MIN_COVAR))
    else:
        covars = np.tile(np.maximum(X.var(axis=0), _MIN_COVAR), (n_regimes, 1))
    return startprob, transmat, means, covars


def _m_step(
    X: FloatArray,
    log_gamma: FloatArray,
    log_xi: FloatArray,
    covariance_type: str,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    gamma = np.clip(np.exp(log_gamma), _EPS, None)
    xi = np.clip(np.exp(log_xi), _EPS, None)
    n_regimes = gamma.shape[1]
    n_features = X.shape[1]

    startprob = gamma[0]
    startprob = startprob / startprob.sum()

    transmat = xi.sum(axis=0)
    transmat = transmat / transmat.sum(axis=1, keepdims=True)
    transmat = np.maximum(transmat, _EPS)
    transmat = transmat / transmat.sum(axis=1, keepdims=True)

    nk = gamma.sum(axis=0) + _EPS
    means = (gamma.T @ X) / nk[:, np.newaxis]

    if covariance_type == "full":
        covars = np.empty((n_regimes, n_features, n_features))
        for k in range(n_regimes):
            diff = X - means[k]
            covars[k] = (diff.T * gamma[:, k]) @ diff / nk[k]
            covars[k].flat[:: n_features + 1] += _MIN_COVAR
    elif covariance_type == "spherical":
        covars = np.empty(n_regimes)
        for k in range(n_regimes):
            diff = X - means[k]
            covars[k] = np.maximum(
                np.sum(gamma[:, k][:, np.newaxis] * diff**2) / (nk[k] * n_features),
                _MIN_COVAR,
            )
    else:
        covars = np.empty((n_regimes, n_features))
        for k in range(n_regimes):
            diff = X - means[k]
            covars[k] = np.maximum(
                np.sum(gamma[:, k][:, np.newaxis] * diff**2, axis=0) / nk[k],
                _MIN_COVAR,
            )
    return startprob, transmat, means, covars


def fit_gaussian_hmm(
    X: FloatArray,
    n_regimes: int,
    covariance_type: str,
    n_iter: int,
    tol: float,
    n_init: int,
    random_state,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, float]:
    """Fit a Gaussian HMM by Baum-Welch, keeping the best of ``n_init`` runs."""
    rng = sku.check_random_state(random_state)
    best: tuple[FloatArray, FloatArray, FloatArray, FloatArray, float] | None = None

    for _init in range(n_init):
        startprob, transmat, means, covars = _init_hmm(
            X,
            n_regimes=n_regimes,
            covariance_type=covariance_type,
            random_state=rng.randint(0, np.iinfo(np.int32).max),
        )
        prev_ll = -np.inf
        log_likelihood = -np.inf
        for _ in range(n_iter):
            log_emit = _log_emissions(X, means, covars, covariance_type)
            log_start = np.log(np.maximum(startprob, _EPS))
            log_trans = np.log(np.maximum(transmat, _EPS))
            log_likelihood, log_gamma, log_xi = _forward_backward(
                log_start, log_trans, log_emit
            )
            if not np.isfinite(log_likelihood):
                break
            startprob, transmat, means, covars = _m_step(
                X, log_gamma, log_xi, covariance_type
            )
            if abs(log_likelihood - prev_ll) < tol:
                break
            prev_ll = log_likelihood
        if best is None or (np.isfinite(log_likelihood) and log_likelihood > best[-1]):
            best = (startprob, transmat, means, covars, float(log_likelihood))

    assert best is not None
    return best


class GaussianHMMDetector(BaseRegimeDetector):
    """Gaussian hidden Markov model for return-regime detection.

    Fits a Gaussian HMM on a low-dimensional summary of asset returns (by
    default the cross-sectional mean and a causal realized-volatility feature)
    and decodes a Viterbi state path. Short runs are merged so that noisy
    flicker does not create spurious change points.

    This estimator is designed to be cloned and fitted **inside each training
    fold** of :class:`~skfolio_regime.RegimeWalkForward`. Fitting on the full
    sample and reusing the labels for every fold leaks future information into
    earlier splits.

    Parameters
    ----------
    n_regimes : int, default=2
        Number of hidden states. Two states (e.g. low/high volatility) is the
        usual industry default.

    feature : {"mean", "vol", "mean_vol", "full"}, default="mean_vol"
        Features built from the return matrix before fitting the HMM.
        ``full`` uses every asset and is only appropriate for small universes.

    covariance_type : {"diag", "full", "spherical"}, default="diag"
        Emission covariance structure.

    n_iter : int, default=50
        Maximum Baum-Welch iterations per random initialization.

    tol : float, default=1e-4
        Convergence tolerance on the log-likelihood.

    n_init : int, default=5
        Number of k-means / EM initializations. The run with the highest
        log-likelihood is kept.

    min_regime_size : int, default=21
        Minimum length of a decoded run. Shorter runs are merged into a
        neighboring regime.

    vol_window : int, default=21
        Rolling window for the realized-volatility feature.

    random_state : int, RandomState instance or None, default=None
        Controls k-means and EM initializations.

    Attributes
    ----------
    labels_ : ndarray of shape (n_observations,)
        In-sample Viterbi labels after short-run merging.

    change_points_ : ndarray of shape (n_change_points,)
        Indices (in ``0 .. n_observations-1``) where ``labels_`` changes.
        Index ``0`` is not included.

    startprob_ : ndarray of shape (n_regimes,)
        Initial state distribution.

    transmat_ : ndarray of shape (n_regimes, n_regimes)
        Transition matrix.

    means_ : ndarray of shape (n_regimes, n_features)
        Emission means in feature space.

    covars_ : ndarray
        Emission covariances. Shape depends on ``covariance_type``.

    log_likelihood_ : float
        Log-likelihood of the selected EM run (on the feature matrix).

    n_features_in_ : int
        Number of assets seen during ``fit``.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during ``fit``. Defined only when ``X`` has
        feature names that are all strings.

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio_regime import GaussianHMMDetector
    >>> rng = np.random.default_rng(0)
    >>> low = rng.normal(0.0, 0.01, size=(80, 3))
    >>> high = rng.normal(0.0, 0.05, size=(80, 3))
    >>> X = np.vstack((low, high))
    >>> det = GaussianHMMDetector(n_regimes=2, feature="vol", random_state=0)
    >>> det.fit(X)
    GaussianHMMDetector(...)
    >>> det.labels_.shape
    (160,)
    """

    def __init__(
        self,
        n_regimes: int = 2,
        feature: Literal["mean", "vol", "mean_vol", "full"] = "mean_vol",
        covariance_type: Literal["diag", "full", "spherical"] = "diag",
        n_iter: int = 50,
        tol: float = 1e-4,
        n_init: int = 5,
        min_regime_size: int = 21,
        vol_window: int = 21,
        random_state: int | None = None,
    ):
        self.n_regimes = n_regimes
        self.feature = feature
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.n_init = n_init
        self.min_regime_size = min_regime_size
        self.vol_window = vol_window
        self.random_state = random_state

    def _validate_params(self) -> None:
        if not isinstance(self.n_regimes, (int, np.integer)) or self.n_regimes < 1:
            raise ValueError(
                f"`n_regimes` must be a positive integer, got {self.n_regimes!r}."
            )
        if self.covariance_type not in {"diag", "full", "spherical"}:
            raise ValueError(
                "`covariance_type` must be 'diag', 'full' or 'spherical', "
                f"got {self.covariance_type!r}."
            )
        if not isinstance(self.n_iter, (int, np.integer)) or self.n_iter < 1:
            raise ValueError(
                f"`n_iter` must be a positive integer, got {self.n_iter!r}."
            )
        if not isinstance(self.n_init, (int, np.integer)) or self.n_init < 1:
            raise ValueError(
                f"`n_init` must be a positive integer, got {self.n_init!r}."
            )
        if (
            not isinstance(self.min_regime_size, (int, np.integer))
            or self.min_regime_size < 1
        ):
            raise ValueError(
                f"`min_regime_size` must be a positive integer, got {self.min_regime_size!r}."
            )
        if not isinstance(self.vol_window, (int, np.integer)) or self.vol_window < 2:
            raise ValueError(
                f"`vol_window` must be an integer >= 2, got {self.vol_window!r}."
            )

    def _features(self, X: ArrayLike) -> FloatArray:
        X_arr = skv.check_array(X, dtype=float, ensure_all_finite=False)
        return extract_regime_features(
            X_arr, feature=self.feature, vol_window=self.vol_window
        )

    def fit(self, X: ArrayLike, y=None):
        """Fit the HMM on ``X`` and decode in-sample regime labels.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Asset returns available at training time.

        y : ignored
            Not used, present for compatibility.

        Returns
        -------
        self : object
            Fitted detector.
        """
        self._validate_params()
        X_arr = skv.validate_data(
            self, X, dtype=float, ensure_all_finite=False, reset=True
        )
        features = extract_regime_features(
            X_arr, feature=self.feature, vol_window=self.vol_window
        )
        n_obs = features.shape[0]
        n_regimes = min(int(self.n_regimes), n_obs)

        startprob, transmat, means, covars, log_likelihood = fit_gaussian_hmm(
            features,
            n_regimes=n_regimes,
            covariance_type=self.covariance_type,
            n_iter=int(self.n_iter),
            tol=float(self.tol),
            n_init=int(self.n_init),
            random_state=self.random_state,
        )
        log_emit = _log_emissions(features, means, covars, self.covariance_type)
        labels = _viterbi(
            np.log(np.maximum(startprob, _EPS)),
            np.log(np.maximum(transmat, _EPS)),
            log_emit,
        )
        labels = merge_short_runs(labels, int(self.min_regime_size))

        self.startprob_ = startprob
        self.transmat_ = transmat
        self.means_ = means
        self.covars_ = covars
        self.log_likelihood_ = log_likelihood
        self.n_regimes_ = n_regimes
        self.labels_ = labels
        self.change_points_ = self._change_points_from_labels(labels)
        return self

    def predict(self, X: ArrayLike) -> IntArray:
        """Decode regime labels on ``X`` with the fitted emission parameters."""
        skv.check_is_fitted(self, "means_")
        features = self._features(X)
        log_emit = _log_emissions(
            features, self.means_, self.covars_, self.covariance_type
        )
        labels = _viterbi(
            np.log(np.maximum(self.startprob_, _EPS)),
            np.log(np.maximum(self.transmat_, _EPS)),
            log_emit,
        )
        return merge_short_runs(labels, int(self.min_regime_size))
