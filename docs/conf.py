"""Configuration file for the Sphinx documentation builder."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "skfolio-regime"
copyright = f"2026-{datetime.now().year}, Carlo Nicolini"
author = "Carlo Nicolini"
release = "0.1.0"
version = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    "sphinxext.opengraph",
    "sphinx_favicon",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "Gemfile",
    "Gemfile.lock",
    "_config.yml",
]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "skfolio-regime"
html_baseurl = "https://carlonicolini.github.io/skfolio-regime/"
html_theme_options = {
    "github_url": "https://github.com/CarloNicolini/skfolio-regime",
    "show_toc_level": 2,
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
}
html_context = {
    "github_user": "CarloNicolini",
    "github_repo": "skfolio-regime",
    "github_version": "main",
    "doc_path": "docs",
}

autosummary_generate = True
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
autodoc_typehints = "none"
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "skfolio": ("https://skfolio.org", None),
}

try:
    from plotly.io._sg_scraper import plotly_sg_scraper

    _image_scrapers = ("matplotlib", plotly_sg_scraper)
except ImportError:
    _image_scrapers = ("matplotlib",)

sphinx_gallery_conf = {
    "doc_module": "skfolio_regime",
    "examples_dirs": str(Path(__file__).resolve().parents[1] / "examples"),
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"plot_",
    "ignore_pattern": r"__init__\.py",
    "backreferences_dir": "generated",
    "reference_url": {"skfolio_regime": None},
    "remove_config_comments": True,
    "plot_gallery": "True",
    "image_scrapers": _image_scrapers,
}

favicons = [{"rel": "icon", "href": "favicon.svg", "type": "image/svg+xml"}]

ogp_site_url = "https://carlonicolini.github.io/skfolio-regime/"
ogp_site_name = "skfolio-regime"


def setup(app):
    css = Path(__file__).parent / "_static" / "css" / "custom.css"
    if css.exists():
        app.add_css_file("css/custom.css")
