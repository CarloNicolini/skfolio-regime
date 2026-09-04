"""Base classes and contract for regime detectors."""

# Copyright (c) 2026
# Author: Carlo Nicolini <c.nicolini@ipazia.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import sklearn as sk
import sklearn.base as skb
import sklearn.utils as sku
import sklearn.utils.validation as skv
from skfolio.typing import ArrayLike, IntArray

__all__ = [
    "BaseRegimeDetector",
    "check_regime_detector",
    "validate_fitted_detector",
]


class BaseRegimeDetector(skb.BaseEstimator, ABC):
    """Contract for a regime / change-point detector.

    :class:`~skfolio_regime.RegimeWalkForward` does not know how regimes are
    defined. It only requires a **scikit-learn estimator** that can be cloned
    and fitted on a training window of asset returns. Any detector — HMM,
    CUSUM, threshold on realized volatility, supervised classifier, external
    labels with a publication lag — is valid if it implements this contract.

    **Signature**

    .. code-block:: python

        class MyDetector(BaseRegimeDetector):
            def __init__(self, *, my_param=1.0):
                self.my_param = my_param

            def fit(self, X, y=None):
                X = self._validate_input(X)
                n_obs = X.shape[0]
                self.labels_ = ...  # ndarray of shape (n_obs,), integer dtype
                return self._finalize_fit(n_obs)

    ``__init__`` must follow scikit-learn: every argument is stored as an
    attribute of the same name, no ``*args`` / ``**kwargs``. That is required
    for :func:`sklearn.base.clone`.

    **Required after ``fit``**

    ``labels_`` : ndarray of shape (n_observations,)
        Integer regime id for each row of ``X``, in ``0 .. n_obs - 1`` order.
        Ids must be non-negative integers. The **absolute** id is not
        identified across folds (label switching); callers use the last
        label as "current regime".

    **Optional after ``fit``**

    ``change_points_`` : ndarray of shape (n_changes,)
        Indices in ``1 .. n_obs - 1`` where ``labels_`` changes. If omitted,
        :meth:`_finalize_fit` computes it.

    ``predict(X)`` : ndarray of shape (n_observations,)
        Optional out-of-sample decoding with parameters learned in ``fit``.
        Not used by :class:`~skfolio_regime.RegimeWalkForward` (the splitter
        only reads in-sample ``labels_``). Implement it if the detector has
        a well-defined filter / Viterbi step.

    **Causality**

    The splitter calls ``clone(detector).fit(X[train])``. ``X`` is already
    truncated to the training window. The detector must not look at rows it
    is not given. Do not fit on the full sample and inject those labels.

    **Minimal custom detector**

    .. code-block:: python

        import numpy as np
        import pandas as pd
        from skfolio_regime import BaseRegimeDetector

        class MedianVolDetector(BaseRegimeDetector):
            '''High/low vol from a causal rolling standard deviation.'''

            def __init__(self, window=21):
                self.window = window

            def fit(self, X, y=None):
                X = self._validate_input(X)
                r = np.nanmean(X, axis=1)
                vol = pd.Series(r).rolling(self.window, min_periods=2).std()
                vol = vol.fillna(0.0).to_numpy()
                self.labels_ = (vol > np.median(vol)).astype(int)
                return self._finalize_fit(X.shape[0])

    Notes
    -----
    Subclass this class rather than relying on duck typing.
    :class:`~skfolio_regime.RegimeWalkForward` checks
    ``isinstance(detector, BaseRegimeDetector)``.
    """

    labels_: IntArray
    change_points_: IntArray
    n_features_in_: int

    @abstractmethod
    def fit(self, X: ArrayLike, y=None):
        """Fit the detector on returns available at training time.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Asset returns. :class:`~skfolio_regime.RegimeWalkForward` passes
            only the current training window.

        y : ignored
            Not used. Present for scikit-learn compatibility. Do not require
            ``y``; regime CV is unsupervised from the splitter's point of view.

        Returns
        -------
        self : object
            Fitted detector. Must set ``labels_``.
        """

    def _validate_input(self, X: ArrayLike, *, reset: bool = True) -> np.ndarray:
        """Validate ``X`` as a 2d return matrix and set ``n_features_in_``."""
        return skv.validate_data(
            self,
            X,
            dtype=float,
            ensure_all_finite=False,
            ensure_2d=True,
            reset=reset,
        )

    def _finalize_fit(self, n_observations: int):
        """Validate ``labels_``, fill ``change_points_``, and return ``self``."""
        validate_fitted_detector(self, n_observations)
        return self

    def predict(self, X: ArrayLike) -> IntArray:
        """Predict regime labels for ``X`` using parameters learned in :meth:`fit`.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Asset returns.

        Returns
        -------
        labels : ndarray of shape (n_observations,)
            Decoded regime labels.

        Raises
        ------
        NotImplementedError
            If the detector only provides in-sample ``labels_``.
        """
        skv.check_is_fitted(self, "labels_")
        raise NotImplementedError(
            f"{type(self).__name__} does not implement `predict`. "
            "In-sample regimes are available as `labels_` after `fit`. "
            "Implement `predict` only if the detector can decode new "
            "observations with parameters frozen at training time."
        )

    @staticmethod
    def change_points_from_labels(labels: IntArray) -> IntArray:
        """Return indices where the label sequence changes.

        Parameters
        ----------
        labels : array-like of shape (n_observations,)
            Integer regime labels.

        Returns
        -------
        change_points : ndarray of shape (n_changes,)
            Indices ``t`` such that ``labels[t] != labels[t - 1]``. ``0`` is
            never included.
        """
        labels = np.asarray(labels)
        if labels.size <= 1:
            return np.array([], dtype=int)
        return np.flatnonzero(np.diff(labels) != 0) + 1

    # Backward-compatible alias used by earlier examples/tests.
    _change_points_from_labels = change_points_from_labels


def validate_fitted_detector(
    estimator: object,
    n_observations: int,
) -> IntArray:
    """Check that a fitted detector satisfies the :class:`BaseRegimeDetector` contract.

    Parameters
    ----------
    estimator : object
        Detector after ``fit``. Must expose ``labels_``.

    n_observations : int
        Number of rows passed to ``fit``.

    Returns
    -------
    labels : ndarray of shape (n_observations,)
        Validated integer labels. Also written back to ``estimator.labels_``.
        ``estimator.change_points_`` is set when missing or overwritten to
        match ``labels_``.

    Raises
    ------
    AttributeError
        If ``labels_`` is missing.

    ValueError
        If ``labels_`` has the wrong shape, contains NaNs, or is not a
        non-negative integer encoding.
    """
    if not hasattr(estimator, "labels_"):
        raise AttributeError(
            f"{type(estimator).__name__} must set `labels_` in `fit` "
            "(1d integer array of length n_observations)."
        )
    labels = np.asarray(estimator.labels_)
    if labels.ndim != 1:
        raise ValueError(
            f"`labels_` must be 1d, got shape {labels.shape} from "
            f"{type(estimator).__name__}."
        )
    if labels.shape[0] != n_observations:
        raise ValueError(
            f"`labels_` length must equal n_observations={n_observations}, "
            f"got {labels.shape[0]} from {type(estimator).__name__}."
        )
    if labels.size == 0:
        raise ValueError("`labels_` is empty.")
    if not np.all(np.isfinite(np.asarray(labels, dtype=float))):
        raise ValueError("`labels_` must be finite.")
    rounded = np.rint(np.asarray(labels, dtype=float))
    if np.any(np.abs(np.asarray(labels, dtype=float) - rounded) > 1e-9):
        raise ValueError("`labels_` must contain integer regime ids.")
    labels_int = rounded.astype(int)
    if np.any(labels_int < 0):
        raise ValueError("`labels_` must be non-negative integer regime ids.")
    estimator.labels_ = np.ascontiguousarray(labels_int)
    estimator.change_points_ = BaseRegimeDetector.change_points_from_labels(
        estimator.labels_
    )
    return estimator.labels_


def check_regime_detector(
    estimator: BaseRegimeDetector,
    X: ArrayLike | None = None,
) -> None:
    """Sanity-check a user detector against the public contract.

    Call this from tests when you implement a custom
    :class:`BaseRegimeDetector`. It clones the estimator, fits it on a small
    return matrix, and checks ``labels_``, ``change_points_``, and
    ``get_params`` / ``clone`` behaviour.

    Parameters
    ----------
    estimator : BaseRegimeDetector
        Unfitted detector instance.

    X : array-like of shape (n_observations, n_assets), optional
        Returns used for the check. A short Gaussian panel is used when
        omitted.

    Raises
    ------
    TypeError
        If ``estimator`` is not a :class:`BaseRegimeDetector`.

    ValueError
        If the fitted attributes violate the contract.
    """
    if not isinstance(estimator, BaseRegimeDetector):
        raise TypeError(
            "Custom detectors must subclass `BaseRegimeDetector`. "
            f"Got {type(estimator).__name__}."
        )
    if X is None:
        rng = np.random.default_rng(0)
        X = rng.normal(scale=0.01, size=(40, 3))
    (X,) = sku.indexable(X)
    n_obs = np.asarray(X).shape[0]

    params = estimator.get_params(deep=False)
    cloned = sk.clone(estimator)
    if cloned is estimator:
        raise ValueError(f"{type(estimator).__name__} is not cloneable.")
    if cloned.get_params(deep=False) != params:
        raise ValueError(
            f"{type(estimator).__name__}.get_params() changed after clone()."
        )

    fitted = cloned.fit(X)
    if fitted is not cloned:
        raise ValueError(
            f"{type(estimator).__name__}.fit must return self, "
            f"got {type(fitted).__name__}."
        )
    validate_fitted_detector(fitted, n_obs)
    if np.max(fitted.change_points_, initial=0) >= n_obs:
        raise ValueError("`change_points_` contains an out-of-range index.")
    if np.any(fitted.change_points_ < 1):
        raise ValueError("`change_points_` must be strictly positive.")
