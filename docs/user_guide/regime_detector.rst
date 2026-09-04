.. _regime_detector:

****************
Regime Detectors
****************

:class:`~skfolio_regime.RegimeWalkForward` is detector-agnostic. A Gaussian
HMM is only the default implementation. Any object that subclasses
:class:`~skfolio_regime.BaseRegimeDetector` can be passed as ``detector=``.

The contract
============

A detector is a scikit-learn estimator:

1. ``__init__`` stores every argument as ``self.arg_name`` (so
   :func:`sklearn.base.clone` works). No ``*args`` / ``**kwargs``.
2. ``fit(X, y=None)`` is called on a **training window** of asset returns
   ``X`` of shape ``(n_observations, n_assets)``.
3. After ``fit``, ``labels_`` is a 1d integer array of length
   ``n_observations``. Entry ``t`` is the regime id of row ``t``.
4. ``fit`` returns ``self``.
5. ``change_points_`` is optional; :meth:`~skfolio_regime.BaseRegimeDetector._finalize_fit`
   fills it from ``labels_``.
6. ``predict(X)`` is optional. The splitter never calls it.

:func:`~skfolio_regime.check_regime_detector` runs these checks on an
unfitted instance. Call it from your tests.

.. code-block:: python

    from skfolio_regime import check_regime_detector
    from my_package import MedianVolDetector

    check_regime_detector(MedianVolDetector(window=21))

Causality
=========

The splitter already truncates ``X`` to the training indices of the current
fold and clones the detector. You must not:

* fit on a longer series than the ``X`` you were given
* reuse labels computed on the full sample
* use ``y`` as a vehicle for future information

Label ids are **not** identified across folds. The splitter treats the last
label of the window as the current regime (``train_scope="current_regime"``).

Minimal implementation
======================

.. code-block:: python

    import numpy as np
    import pandas as pd

    from skfolio_regime import BaseRegimeDetector, RegimeWalkForward

    class MedianVolDetector(BaseRegimeDetector):
        """High vs low volatility from a causal rolling standard deviation."""

        def __init__(self, window=21):
            self.window = window

        def fit(self, X, y=None):
            X = self._validate_input(X)
            r = np.nanmean(X, axis=1)
            vol = (
                pd.Series(r)
                .rolling(self.window, min_periods=2)
                .std()
                .fillna(0.0)
                .to_numpy()
            )
            self.labels_ = (vol > np.median(vol)).astype(int)
            return self._finalize_fit(X.shape[0])

    cv = RegimeWalkForward(
        test_size=21,
        train_size=252,
        detector=MedianVolDetector(window=21),
        train_scope="current_regime",
    )

Helpers on the base class
=========================

* :meth:`~skfolio_regime.BaseRegimeDetector._validate_input` — 2d float
  ``X``, sets ``n_features_in_``.
* :meth:`~skfolio_regime.BaseRegimeDetector._finalize_fit` — validates
  ``labels_`` and writes ``change_points_``.
* :func:`~skfolio_regime.validate_fitted_detector` — the same checks,
  usable outside ``fit``.

The bundled Gaussian HMM
========================

:class:`~skfolio_regime.GaussianHMMDetector` is one implementation of this
contract: a Gaussian HMM on a low-dimensional summary of returns. You can
replace it without changing the splitter.
