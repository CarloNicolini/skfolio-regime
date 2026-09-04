"""Tests for RegimeWalkForward."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import InverseVolatility
from skfolio.pre_selection import SelectKExtremes
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from skfolio_regime import GaussianHMMDetector, RegimeWalkForward
from tests.helpers import FixedLabelsDetector


def test_is_walk_forward_subclass():
    cv = RegimeWalkForward(test_size=2, train_size=5, detector=FixedLabelsDetector(3))
    assert isinstance(cv, WalkForward)


def test_current_regime_keeps_suffix():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    cv = RegimeWalkForward(
        test_size=5,
        train_size=12,
        detector=FixedLabelsDetector(n_current=4),
        train_scope="current_regime",
        min_train_size=1,
    )
    train, test = next(cv.split(X))
    np.testing.assert_array_equal(train, np.arange(8, 12))
    np.testing.assert_array_equal(test, np.arange(12, 17))
    assert train.max() < test.min()


def test_same_regime_keeps_all_matching_labels():
    X = np.random.default_rng(0).normal(size=(25, 2))
    cv = RegimeWalkForward(
        test_size=3,
        train_size=10,
        detector=FixedLabelsDetector(n_current=4),
        train_scope="same_regime",
        min_train_size=1,
    )
    train, _test = next(cv.split(X))
    np.testing.assert_array_equal(train, np.arange(6, 10))


def test_all_past_keeps_walk_forward_train():
    X = np.random.default_rng(0).normal(size=(25, 2))
    base = WalkForward(test_size=3, train_size=10)
    cv = RegimeWalkForward(
        test_size=3,
        train_size=10,
        detector=FixedLabelsDetector(n_current=3),
        train_scope="all_past",
        min_train_size=1,
    )
    expected = list(base.split(X))
    got = list(cv.split(X))
    assert len(got) == len(expected)
    for (tr, te), (etr, ete) in zip(got, expected, strict=True):
        np.testing.assert_array_equal(tr, etr)
        np.testing.assert_array_equal(te, ete)


def test_purged_size_gap():
    X = np.random.default_rng(0).normal(size=(40, 2))
    cv = RegimeWalkForward(
        test_size=4,
        train_size=10,
        purged_size=2,
        detector=FixedLabelsDetector(5),
        min_train_size=1,
    )
    train, test = next(cv.split(X))
    assert test.min() - train.max() == 3  # 2 purged observations between them


def test_detector_never_sees_test_rows():
    X = np.arange(40, dtype=float).reshape(-1, 1)  # row i has unique value i

    class CaptureDetector(FixedLabelsDetector):
        def fit(self, X, y=None):
            super().fit(X, y)
            self.seen_max_ = float(np.nanmax(np.asarray(X)))
            return self

    cv = RegimeWalkForward(
        test_size=5,
        train_size=10,
        detector=CaptureDetector(4),
        min_train_size=1,
    )
    seen = []
    for train, test in cv.split(X):
        det = CaptureDetector(4)
        det.fit(X[train])
        seen.append((det.seen_max_, train.max(), test.min()))
        assert det.seen_max_ == float(train.max())
        assert train.max() < test.min()
        assert np.intersect1d(train, test).size == 0


def test_short_train_fallback_preserves_n_splits():
    X = np.random.default_rng(0).normal(size=(40, 2))
    base = WalkForward(test_size=5, train_size=12)
    cv = RegimeWalkForward(
        test_size=5,
        train_size=12,
        detector=FixedLabelsDetector(n_current=2),
        min_train_size=8,
        short_train="fallback",
        train_scope="current_regime",
    )
    assert cv.get_n_splits(X) == base.get_n_splits(X)
    assert len(list(cv.split(X))) == base.get_n_splits(X)


def test_short_train_skip_drops_early_folds():
    X = np.random.default_rng(0).normal(size=(60, 2))

    class LastHalfDetector(FixedLabelsDetector):
        def fit(self, X, y=None):
            n_obs = np.asarray(X).shape[0]
            self.n_current = max(1, n_obs // 2)
            return super().fit(X, y)

    cv = RegimeWalkForward(
        test_size=5,
        train_size=10,
        expand_train=True,
        detector=LastHalfDetector(n_current=1),
        min_train_size=18,
        short_train="skip",
        train_scope="current_regime",
    )
    splits = list(cv.split(X))
    assert cv.n_skipped_ > 0
    assert splits
    assert len(splits) == cv.get_n_splits(X)
    base_n = WalkForward(test_size=5, train_size=10, expand_train=True).get_n_splits(X)
    assert len(splits) < base_n


def test_confirmation_delay_requires_stable_tail():
    X = np.random.default_rng(0).normal(size=(30, 2))

    class FlickerDetector(FixedLabelsDetector):
        def fit(self, X, y=None):
            super().fit(X, y)
            labels = np.zeros(X.shape[0], dtype=int)
            labels[-1] = 1
            self.labels_ = labels
            self.change_points_ = self._change_points_from_labels(labels)
            return self

    cv = RegimeWalkForward(
        test_size=5,
        train_size=10,
        detector=FlickerDetector(1),
        confirmation_delay=3,
        min_train_size=1,
        short_train="fallback",
        train_scope="current_regime",
    )
    train, _test = next(cv.split(X))
    np.testing.assert_array_equal(train, np.arange(10))


def test_expand_train_still_causal():
    X = np.random.default_rng(0).normal(size=(35, 2))
    cv = RegimeWalkForward(
        test_size=5,
        train_size=10,
        expand_train=True,
        detector=FixedLabelsDetector(n_current=6),
        train_scope="all_past",
        min_train_size=1,
    )
    prev_test_end = -1
    for train, test in cv.split(X):
        assert train.min() == 0
        assert train.max() < test.min()
        assert test.min() > prev_test_end
        prev_test_end = test.max()


def test_invalid_train_scope():
    cv = RegimeWalkForward(
        test_size=2, train_size=5, train_scope="oracle", detector=FixedLabelsDetector()
    )
    with pytest.raises(ValueError, match="train_scope"):
        list(cv.split(np.zeros((20, 2))))


def test_invalid_detector_type():
    cv = RegimeWalkForward(test_size=2, train_size=5, detector=WalkForward(2, 5))
    with pytest.raises(TypeError, match="BaseRegimeDetector"):
        list(cv.split(np.zeros((20, 2))))


def test_cross_val_predict_and_grid_search():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(80, 6)), columns=list("ABCDEF"))
    cv = RegimeWalkForward(
        test_size=10,
        train_size=25,
        detector=FixedLabelsDetector(n_current=12),
        min_train_size=5,
    )
    model = Pipeline(
        [
            ("pre_selection", SelectKExtremes(k=4)),
            ("allocation", InverseVolatility()),
        ]
    )
    pred = cross_val_predict(model, X, cv=cv)
    assert len(pred) == cv.get_n_splits(X)

    gs = GridSearchCV(
        estimator=model,
        cv=cv,
        param_grid={"pre_selection__k": [2, 3]},
    )
    gs.fit(X)
    assert gs.best_estimator_ is not None


def test_hmm_walk_forward_on_synthetic_returns():
    rng = np.random.default_rng(2)
    low = rng.normal(0.0, 0.01, size=(60, 4))
    high = rng.normal(0.0, 0.05, size=(60, 4))
    X = np.vstack((low, high))
    cv = RegimeWalkForward(
        test_size=10,
        train_size=40,
        detector=GaussianHMMDetector(
            n_regimes=2, feature="vol", min_regime_size=8, random_state=0
        ),
        min_train_size=8,
    )
    splits = list(cv.split(X))
    assert splits
    for train, test in splits:
        assert train.size >= 8
        assert train.max() < test.min()
        assert np.all(np.diff(train) > 0)
        assert np.all(np.diff(test) > 0)


def test_get_n_splits_requires_x_when_skipping():
    cv = RegimeWalkForward(
        test_size=5,
        train_size=10,
        short_train="skip",
        detector=FixedLabelsDetector(2),
        min_train_size=8,
    )
    with pytest.raises(ValueError, match="X"):
        cv.get_n_splits()
