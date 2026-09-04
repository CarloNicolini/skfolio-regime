"""Regime-aware Walk-Forward cross-validator."""

# Copyright (c) 2026
# Author: Carlo Nicolini <c.nicolini@ipazia.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from typing import Literal

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.utils as sku
from skfolio.model_selection import WalkForward
from skfolio.typing import ArrayLike, IntArray
from skfolio.utils.tools import safe_split

from skfolio_regime._base import BaseRegimeDetector, validate_fitted_detector
from skfolio_regime._gaussian_hmm import GaussianHMMDetector

_TRAIN_SCOPES = {"all_past", "current_regime", "same_regime"}
_SHORT_TRAIN = {"fallback", "skip"}


class RegimeWalkForward(WalkForward):
    """Walk-Forward cross-validator with causal regime-filtered training sets.

    Fold **boundaries** are those of :class:`~skfolio.model_selection.WalkForward`
    (observation counts or calendar frequency, with optional purging). For each
    fold the detector is **cloned and fitted only on** ``X[train]``. Test
    observations are never used to infer regimes, so fold placement stays
    temporally valid.

    Parameters
    ----------
    test_size : int
        Length of each test set. If ``freq`` is ``None`` (default), it is a
        number of observations. Otherwise it is a number of periods defined by
        ``freq``.

    train_size : int | pandas.offsets.DateOffset | datetime.timedelta
        Length of each WalkForward training window, before regime filtering.

    detector : BaseRegimeDetector, optional
        A custom :class:`~skfolio_regime.BaseRegimeDetector`. Must be
        cloneable (scikit-learn ``__init__`` contract) and set integer
        ``labels_`` of length ``len(train)`` in ``fit``. See
        :func:`~skfolio_regime.check_regime_detector`. The default
        (``None``) uses :class:`~skfolio_regime.GaussianHMMDetector` with
        two states.

    train_scope : {"current_regime", "same_regime", "all_past"}, default="current_regime"
        How training indices are filtered after decoding the train window:

        * ``current_regime``: keep the last contiguous run of the current
          (end-of-train) state.
        * ``same_regime``: keep every observation labeled as the current state
          (possibly non-contiguous).
        * ``all_past``: keep the original WalkForward training indices
          (detector is then only stored for diagnostics).

    min_train_size : int, optional
        Minimum number of observations required after filtering. If ``None``,
        defaults to ``min(63, len(train_wf))`` for integer ``train_size`` and
        to ``1`` otherwise.

    confirmation_delay : int, default=0
        Require the last ``confirmation_delay`` training labels to be identical
        before trusting the current regime. If the label flickers, the fold is
        treated as a short training set (see ``short_train``).

    short_train : {"fallback", "skip"}, default="fallback"
        Action when the filtered training set is too short or the regime is
        unconfirmed. ``fallback`` restores the WalkForward training window so
        that the number of splits matches the parent splitter. ``skip`` drops
        the fold.

    freq : str | pandas.offsets.DateOffset, optional
        Calendar frequency forwarded to :class:`~skfolio.model_selection.WalkForward`.

    freq_offset : pandas DateOffset | datetime timedelta, optional
        Offset applied to ``freq``.

    previous : bool, default=False
        Alignment of calendar dates that are missing from the index.

    expand_train : bool, default=False
        If ``True``, the WalkForward training window is expanding (from the
        first observation). Regime filtering is still applied to that window.

    reduce_test : bool, default=False
        Keep a partial last test window.

    purged_size : int, default=0
        Observations dropped between the end of training and the start of
        test. See :class:`~skfolio.model_selection.WalkForward`.

    Attributes
    ----------
    n_skipped_ : int
        Number of folds skipped during the last call to :meth:`split`.
        Defined after :meth:`split`.

    last_labels_ : ndarray
        Regime labels of the last yielded training window, aligned with the
        unfiltered WalkForward train indices. Defined after :meth:`split`.

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio_regime import GaussianHMMDetector, RegimeWalkForward
    >>> X = np.random.default_rng(0).normal(size=(80, 2))
    >>> cv = RegimeWalkForward(
    ...     test_size=5,
    ...     train_size=20,
    ...     detector=GaussianHMMDetector(n_regimes=2, min_regime_size=5, random_state=0),
    ... )
    >>> splits = list(cv.split(X))
    >>> train, test = splits[0]
    >>> train.max() < test.min()
    True

    See Also
    --------
    GaussianHMMDetector : Default two-state Gaussian HMM detector.
    skfolio.model_selection.WalkForward : Parent splitter (rebalancing grid).
    """

    def __init__(
        self,
        test_size: int,
        train_size: int | pd.offsets.BaseOffset | dt.timedelta,
        detector: BaseRegimeDetector | None = None,
        train_scope: Literal[
            "current_regime", "same_regime", "all_past"
        ] = "current_regime",
        min_train_size: int | None = None,
        confirmation_delay: int = 0,
        short_train: Literal["fallback", "skip"] = "fallback",
        freq: str | pd.offsets.BaseOffset | None = None,
        freq_offset: pd.offsets.BaseOffset | dt.timedelta | None = None,
        previous: bool = False,
        expand_train: bool = False,
        reduce_test: bool = False,
        purged_size: int = 0,
    ):
        super().__init__(
            test_size=test_size,
            train_size=train_size,
            freq=freq,
            freq_offset=freq_offset,
            previous=previous,
            expand_train=expand_train,
            reduce_test=reduce_test,
            purged_size=purged_size,
        )
        self.detector = detector
        self.train_scope = train_scope
        self.min_train_size = min_train_size
        self.confirmation_delay = confirmation_delay
        self.short_train = short_train

    def _resolved_detector(self) -> BaseRegimeDetector:
        if self.detector is None:
            return GaussianHMMDetector()
        if not isinstance(self.detector, BaseRegimeDetector):
            raise TypeError(
                "`detector` must be a `BaseRegimeDetector` instance or None, "
                f"got {type(self.detector)!r}."
            )
        return self.detector

    def _validate_regime_params(self) -> None:
        if self.train_scope not in _TRAIN_SCOPES:
            raise ValueError(
                f"`train_scope` must be one of {sorted(_TRAIN_SCOPES)}, "
                f"got {self.train_scope!r}."
            )
        if self.short_train not in _SHORT_TRAIN:
            raise ValueError(
                f"`short_train` must be one of {sorted(_SHORT_TRAIN)}, "
                f"got {self.short_train!r}."
            )
        if (
            not isinstance(self.confirmation_delay, (int, np.integer))
            or self.confirmation_delay < 0
        ):
            raise ValueError(
                "`confirmation_delay` must be a non-negative integer, "
                f"got {self.confirmation_delay!r}."
            )
        if self.min_train_size is not None and (
            not isinstance(self.min_train_size, (int, np.integer))
            or self.min_train_size < 1
        ):
            raise ValueError(
                f"`min_train_size` must be a positive integer or None, "
                f"got {self.min_train_size!r}."
            )

    def _min_train_size(self, n_train_wf: int) -> int:
        if self.min_train_size is not None:
            return int(self.min_train_size)
        if isinstance(self.train_size, (int, np.integer)):
            return min(63, n_train_wf)
        return 1

    def _filter_train(
        self, train_wf: IntArray, labels: IntArray
    ) -> tuple[IntArray, bool]:
        """Return filtered train indices and whether the regime was confirmed."""
        labels = np.asarray(labels)
        if labels.shape[0] != train_wf.shape[0]:
            raise ValueError(
                "Detector `labels_` length must match the training window, "
                f"got {labels.shape[0]} labels for {train_wf.shape[0]} observations."
            )
        if self.train_scope == "all_past":
            return train_wf, True

        delay = int(self.confirmation_delay)
        if delay > 0:
            if labels.size < delay:
                return train_wf, False
            tail = labels[-delay:]
            if np.any(tail != tail[-1]):
                return train_wf, False
            current = int(tail[-1])
        else:
            current = int(labels[-1])

        if self.train_scope == "same_regime":
            mask = labels == current
            return train_wf[mask], True

        # current_regime: last contiguous run of `current`.
        changes = np.flatnonzero(labels != current)
        start = 0 if changes.size == 0 else int(changes[-1]) + 1
        return train_wf[start:], True

    def split(
        self, X: ArrayLike, y=None, groups=None
    ) -> Iterator[tuple[IntArray, IntArray]]:
        """Generate causally regime-filtered train/test indices.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_targets)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_observations,)
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            Training set indices for that split.

        test : ndarray
            Testing set indices for that split.
        """
        self._validate_regime_params()
        detector = self._resolved_detector()
        X, y = sku.indexable(X, y)

        self.n_skipped_ = 0
        n_yielded = 0
        last_labels: np.ndarray | None = None

        for train_wf, test in super().split(X, y, groups):
            X_train, _ = safe_split(X, y, indices=train_wf, axis=0)
            fitted = sk.clone(detector)
            fitted.fit(X_train)
            last_labels = validate_fitted_detector(fitted, n_observations=len(train_wf))
            train, confirmed = self._filter_train(train_wf, last_labels)
            min_size = self._min_train_size(len(train_wf))
            too_short = train.size < min_size
            if (not confirmed) or too_short:
                if self.short_train == "skip":
                    self.n_skipped_ += 1
                    continue
                train = train_wf
            n_yielded += 1
            yield np.asarray(train, dtype=int), np.asarray(test, dtype=int)

        if last_labels is not None:
            self.last_labels_ = last_labels
        self._n_splits_ = n_yielded

        if n_yielded == 0:
            raise ValueError(
                "RegimeWalkForward produced no splits. Relax `min_train_size`, "
                "`min_regime_size`, or set `short_train='fallback'`."
            )

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations in the cross-validator.

        When ``short_train='fallback'`` this matches
        :class:`~skfolio.model_selection.WalkForward`. When folds can be
        skipped, the detector is run on ``X`` (same procedure as :meth:`split`).

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_targets)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_observations,)
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Number of splitting iterations.
        """
        self._validate_regime_params()
        if self.short_train == "fallback":
            return super().get_n_splits(X=X, y=y, groups=groups)
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        return sum(1 for _ in self.split(X, y, groups))
