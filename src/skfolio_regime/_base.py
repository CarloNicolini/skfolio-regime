"""Base classes for regime detectors."""

# Copyright (c) 2026
# Author: Carlo Nicolini <c.nicolini@ipazia.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import sklearn.base as skb
import sklearn.utils.validation as skv
from skfolio.typing import ArrayLike, IntArray


class BaseRegimeDetector(skb.BaseEstimator, ABC):
    """Base class for regime / change-point detectors.

    Implementations must define :meth:`fit`. After fitting, the estimator exposes
    ``labels_`` (one integer state per observation) and ``change_points_`` (start
    indices of each new regime, excluding 0).

    Notes
    -----
    All estimators should specify all parameters as explicit keyword arguments in
    ``__init__`` (no ``*args`` or ``**kwargs``), following scikit-learn
    conventions.
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
            Asset returns. Callers must pass only observations that would have
            been known at the fold's training cutoff.

        y : ignored
            Not used, present for scikit-learn compatibility.

        Returns
        -------
        self : object
            Fitted detector.
        """

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
        """
        skv.check_is_fitted(self, "labels_")
        raise NotImplementedError(
            f"{type(self).__name__} does not implement `predict`. Use `labels_` "
            "from `fit` for in-sample decoding."
        )

    @staticmethod
    def _change_points_from_labels(labels: IntArray) -> IntArray:
        """Return indices where the label sequence changes."""
        if labels.size <= 1:
            return np.array([], dtype=int)
        return np.flatnonzero(np.diff(labels) != 0) + 1
