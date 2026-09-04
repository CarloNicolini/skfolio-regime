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

The HTML lands in ``docs/_build/html``. The GitHub Actions workflow
``.github/workflows/docs.yml`` builds the same Sphinx site, runs Jekyll with
``baseurl: /skfolio-regime``, and publishes
https://carlonicolini.github.io/skfolio-regime/ .

To preview the Jekyll step locally (Ruby 3 + Bundler):

.. code-block:: bash

    make -C docs html
    mkdir -p jekyll_src
    cp -a docs/_build/html/. jekyll_src/
    cp docs/_config.yml docs/Gemfile jekyll_src/
    touch jekyll_src/.nojekyll
    cd jekyll_src
    bundle install
    bundle exec jekyll serve --baseurl /skfolio-regime

