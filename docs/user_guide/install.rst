.. _install:

============
Installation
============

`skfolio-regime` requires Python 3.10 or later and ``skfolio>=1.0.0``.

Using uv
========

This project is developed with `Astral uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

    git clone https://github.com/CarloNicolini/skfolio-regime.git
    cd skfolio-regime
    uv sync
    source .venv/bin/activate

Using pip
=========

.. code-block:: bash

    pip install -e ".[dev,docs]"

Building the documentation
==========================

.. code-block:: bash

    source .venv/bin/activate
    make -C docs html
