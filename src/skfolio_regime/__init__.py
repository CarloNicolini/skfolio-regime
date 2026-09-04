"""Regime-aware cross-validation for skfolio."""

# Copyright (c) 2026
# Author: Carlo Nicolini <c.nicolini@ipazia.com>
# SPDX-License-Identifier: BSD-3-Clause

from importlib.metadata import PackageNotFoundError, version

from skfolio_regime._base import (
    BaseRegimeDetector,
    check_regime_detector,
    validate_fitted_detector,
)
from skfolio_regime._gaussian_hmm import (
    GaussianHMMDetector,
    extract_regime_features,
    standardize_features,
)
from skfolio_regime._regime_walk_forward import RegimeWalkForward

try:
    __version__ = version("skfolio-regime")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "BaseRegimeDetector",
    "GaussianHMMDetector",
    "RegimeWalkForward",
    "check_regime_detector",
    "extract_regime_features",
    "standardize_features",
    "validate_fitted_detector",
]
