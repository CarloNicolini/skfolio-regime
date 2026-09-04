"""Shared test detectors."""

from __future__ import annotations

import numpy as np
from skfolio.typing import ArrayLike

from skfolio_regime import BaseRegimeDetector


class FixedLabelsDetector(BaseRegimeDetector):
    """Assigns label ``1`` to the last ``n_current`` training observations."""

    def __init__(self, n_current: int = 10):
        self.n_current = n_current

    def fit(self, X: ArrayLike, y=None):
        X_arr = self._validate_input(X)
        n_obs = X_arr.shape[0]
        n_current = min(int(self.n_current), n_obs)
        labels = np.zeros(n_obs, dtype=int)
        if n_current:
            labels[-n_current:] = 1
        self.labels_ = labels
        return self._finalize_fit(n_obs)


class MedianVolDetector(BaseRegimeDetector):
    """Two-state detector from a causal rolling standard deviation.

    Example of a user-written detector that is not an HMM.
    """

    def __init__(self, window: int = 10):
        self.window = window

    def fit(self, X: ArrayLike, y=None):
        X_arr = self._validate_input(X)
        r = np.mean(X_arr, axis=1)
        n_obs = r.size
        vol = np.empty(n_obs)
        w = int(self.window)
        for t in range(n_obs):
            start = max(0, t - w + 1)
            sl = r[start : t + 1]
            vol[t] = float(np.std(sl)) if sl.size >= 2 else 0.0
        self.labels_ = (vol > np.median(vol)).astype(int)
        return self._finalize_fit(n_obs)
