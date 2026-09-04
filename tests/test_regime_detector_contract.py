"""Tests for the public regime-detector contract."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from skfolio_regime import (
    BaseRegimeDetector,
    GaussianHMMDetector,
    RegimeWalkForward,
    check_regime_detector,
    validate_fitted_detector,
)
from tests.helpers import FixedLabelsDetector, MedianVolDetector


def test_check_regime_detector_accepts_custom_and_hmm():
    check_regime_detector(FixedLabelsDetector(n_current=5))
    check_regime_detector(MedianVolDetector(window=8))
    check_regime_detector(
        GaussianHMMDetector(n_regimes=2, min_regime_size=5, n_init=2, random_state=0)
    )


def test_check_regime_detector_rejects_plain_object():
    class NotADetector:
        def fit(self, X, y=None):
            return self

    with pytest.raises(TypeError, match="BaseRegimeDetector"):
        check_regime_detector(NotADetector())


def test_validate_fitted_detector_rejects_bad_labels():
    class BadLength(BaseRegimeDetector):
        def fit(self, X, y=None):
            X = self._validate_input(X)
            self.labels_ = np.array([0, 1])
            return self

    est = BadLength().fit(np.zeros((10, 2)))
    with pytest.raises(ValueError, match="n_observations"):
        validate_fitted_detector(est, n_observations=10)


def test_validate_fitted_detector_rejects_negative_and_nan():
    class Negative(BaseRegimeDetector):
        def __init__(self, labels=None):
            self.labels = labels

        def fit(self, X, y=None):
            X = self._validate_input(X)
            self.labels_ = np.asarray(self.labels)
            return self

    X = np.zeros((4, 2))
    with pytest.raises(ValueError, match="non-negative"):
        validate_fitted_detector(Negative(labels=[-1, 0, 0, 0]).fit(X), 4)
    with pytest.raises(ValueError, match="finite"):
        validate_fitted_detector(Negative(labels=[0, np.nan, 0, 1]).fit(X), 4)


def test_custom_detector_in_regime_walk_forward():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    cv = RegimeWalkForward(
        test_size=5,
        train_size=12,
        detector=MedianVolDetector(window=6),
        min_train_size=4,
    )
    splits = list(cv.split(X))
    assert splits
    for train, test in splits:
        assert train.max() < test.min()


def test_clone_does_not_share_fitted_state():
    X = np.random.default_rng(0).normal(size=(30, 2))
    det = MedianVolDetector(window=5)
    fitted = clone(det).fit(X)
    fresh = clone(fitted)
    assert not hasattr(fresh, "labels_")
