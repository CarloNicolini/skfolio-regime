==============
skfolio-regime
==============

.. image:: https://github.com/CarloNicolini/skfolio-regime/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/CarloNicolini/skfolio-regime/actions/workflows/ci.yml
   :alt: CI

.. image:: https://img.shields.io/badge/docs-GitHub%20Pages-blue
   :target: https://carlonicolini.github.io/skfolio-regime/
   :alt: Documentation

.. skfolio-regime-shared-introduction-start

Regime-aware walk-forward cross-validation for `skfolio`_.

`skfolio-regime` adds a :class:`~skfolio_regime.GaussianHMMDetector` and a
:class:`~skfolio_regime.RegimeWalkForward` splitter that fit naturally into
``skfolio.model_selection``: they subclass :class:`~skfolio.model_selection.WalkForward`,
yield standard ``(train, test)`` indices, and remain compatible with
:func:`~skfolio.model_selection.cross_val_predict`, :class:`~sklearn.pipeline.Pipeline`,
and :class:`~sklearn.model_selection.GridSearchCV`.

Regime labels are inferred **inside each training fold** so that change points
are never estimated from observations that would be unavailable at rebalance
time.

.. _skfolio: https://skfolio.org

.. skfolio-regime-shared-introduction-end

.. skfolio-regime-shared-body-start

Installation
============

Requires Python 3.10+ and ``skfolio>=1.0.0``. Using `uv`_:

.. code-block:: bash

    uv sync
    source .venv/bin/activate

Or with pip:

.. code-block:: bash

    pip install -e ".[dev]"

Quick start
===========

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset
    from skfolio.model_selection import cross_val_predict
    from skfolio.optimization import InverseVolatility
    from skfolio.preprocessing import prices_to_returns

    from skfolio_regime import GaussianHMMDetector, RegimeWalkForward

    X = prices_to_returns(load_sp500_dataset())
    cv = RegimeWalkForward(
        test_size=21,
        train_size=252,
        purged_size=1,
        detector=GaussianHMMDetector(n_regimes=2, random_state=0),
        train_scope="current_regime",
        min_train_size=63,
    )
    pred = cross_val_predict(InverseVolatility(), X, cv=cv)

Documentation
=============

The user guide is published at
https://carlonicolini.github.io/skfolio-regime/

Build it locally:

.. code-block:: bash

    source .venv/bin/activate
    make -C docs html

GitHub Actions runs Sphinx and then Jekyll (``docs/_config.yml``,
``baseurl: /skfolio-regime``) and deploys to GitHub Pages. Enable
**Settings → Pages → Source: GitHub Actions** on the repository.

.. _uv: https://docs.astral.sh/uv/

.. skfolio-regime-shared-body-end
