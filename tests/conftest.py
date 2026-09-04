"""conftest module."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def random_returns():
    rng = np.random.default_rng(42)
    return rng.normal(size=(80, 4))
