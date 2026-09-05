.. _model_selection:

***************
Model Selection
***************

The :class:`~skfolio_regime.RegimeWalkForward` splitter is a
:class:`~skfolio.model_selection.WalkForward` subclass. Fold **when** (rebalancing
grid, purge, expanding vs rolling train) is unchanged. Fold **what** (which
training observations are fed to the portfolio estimator) is filtered using a
regime detector fitted **only on that training window**.

This preserves the guarantees already provided by ``skfolio``:

* ``split`` yields ``(train, test)`` index arrays, so
  :func:`~skfolio.model_selection.cross_val_predict` and
  :class:`~sklearn.model_selection.GridSearchCV` work without a new CV family.
* Tests are strictly after the training cutoff, with optional ``purged_size``.
* Because the class inherits from :class:`~skfolio.model_selection.WalkForward`,
  sequential ``previous_weights`` propagation in ``cross_val_predict`` still
  applies, and :class:`~skfolio.model_selection.MultipleRandomizedCV` can wrap
  it as its inner walk-forward.

.. danger::

    Never fit a detector on the full sample and reuse those change points for
    every fold. That looks ahead: later returns would influence earlier
    training sets. :class:`~skfolio_regime.RegimeWalkForward` clones the
    detector and calls ``fit`` on ``X[train]`` of each fold.

Causal protocol
===============

For each WalkForward pair ``(train_wf, test)``:

1. Clone ``detector``.
2. Fit it on ``X[train_wf]`` only.
3. Read ``labels_`` and optionally require a stable tail of length
   ``confirmation_delay``.
4. Filter ``train_wf`` according to ``train_scope``.
5. If the filtered set is shorter than ``min_train_size``, either restore
   ``train_wf`` (``short_train="fallback"``, default) or skip the fold.

The test indices are never passed to the detector.
When ``short_train="skip"``, set every detector ``random_state``. Otherwise
separate calls to ``get_n_splits`` and ``split`` can accept different folds,
so the splitter raises a ``ValueError`` instead of returning an inconsistent
split count.

Writing your own detector
=========================

``detector`` can be any :class:`~skfolio_regime.BaseRegimeDetector`. See
:ref:`regime_detector` for the signature, the required ``labels_`` contract,
and a complete non-HMM example. Use
:func:`~skfolio_regime.check_regime_detector` in tests to validate a custom
class.

Train scopes
============

* ``current_regime`` (default): last contiguous run of the state at the end of
  the training window. This is the usual "fit on the current HMM regime"
  backtest.
* ``same_regime``: every observation in the window with that state label
  (possibly non-contiguous).
* ``all_past``: no filtering; useful as a control that still records labels.

Gaussian HMM
============

:class:`~skfolio_regime.GaussianHMMDetector` implements a two-state Gaussian
HMM on a low-dimensional summary of returns (cross-sectional mean and/or
causal realized volatility). Full-asset emissions (``feature="full"``) are
supported for small universes but are not the default: rolling covariance HMMs
are poorly determined on typical equity panels.

Short decoded runs are merged via ``min_regime_size`` so that EM flicker does
not explode the number of apparent breaks.

Example
=======

.. code-block:: python

    from skfolio.model_selection import cross_val_predict
    from skfolio.optimization import InverseVolatility

    from skfolio_regime import GaussianHMMDetector, RegimeWalkForward

    cv = RegimeWalkForward(
        test_size=21,
        train_size=252,
        purged_size=1,
        detector=GaussianHMMDetector(n_regimes=2, random_state=0),
        train_scope="current_regime",
        min_train_size=63,
    )
    pred = cross_val_predict(InverseVolatility(), X, cv=cv)
