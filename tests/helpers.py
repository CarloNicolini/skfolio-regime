"""Shared test detectors."""

from __future__ import annotations

import numpy as np
import sklearn.utils.validation as skv
from skfolio.typing import ArrayLike

from skfolio_regime import BaseRegimeDetector


class FixedLabelsDetector(BaseRegimeDetector):
    """Assigns label ``1`` to the last ``n_current`` training observations."""

    def __init__(self, n_current: int = 10):
        self.n_current = n_current

    def fit(self, X: ArrayLike, y=None):
        X_arr = skv.validate_data(self, X, dtype=float, reset=True)
        n_obs = X_arr.shape[0]
        n_current = min(int(self.n_current), n_obs)
        labels = np.zeros(n_obs, dtype=int)
        if n_current:
            labels[-n_current:] = 1
        self.labels_ = labels
        self.change_points_ = self._change_points_from_labels(labels)
        self.n_observations_seen_ = n_obs
        return self
