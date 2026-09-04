"""Regime-aware cross-validation for skfolio."""

# Copyright (c) 2026
# Author: Carlo Nicolini <c.nicolini@ipazia.com>
# SPDX-License-Identifier: BSD-3-Clause

from skfolio_regime._base import BaseRegimeDetector
from skfolio_regime._gaussian_hmm import GaussianHMMDetector, extract_regime_features
from skfolio_regime._regime_walk_forward import RegimeWalkForward

__all__ = [
    "BaseRegimeDetector",
    "GaussianHMMDetector",
    "RegimeWalkForward",
    "extract_regime_features",
]
