"""
===========================
Regime Walk-Forward CV
===========================

This tutorial shows :class:`~skfolio_regime.RegimeWalkForward`, a
:class:`~skfolio.model_selection.WalkForward` subclass that filters each
training window with a causally fitted Gaussian HMM.

Fold dates follow the usual walk-forward grid. The HMM is cloned and fitted on
``X[train]`` only, so regime labels never use observations from the test
period.
"""

# %%
# Data
# ====
from plotly.io import show
from skfolio import Population
from skfolio.datasets import load_sp500_dataset
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import InverseVolatility
from skfolio.preprocessing import prices_to_returns
from sklearn.model_selection import train_test_split

from skfolio_regime import GaussianHMMDetector, RegimeWalkForward

prices = load_sp500_dataset()
X = prices_to_returns(prices)
X = X.loc["2018":"2022", X.columns[:12]]
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# %%
# Baseline walk-forward
# =====================
# Monthly-style rebalancing: 21 test days after 252 training days, with a
# one-observation purge so decisions made at :math:`t` affect returns from
# :math:`t+1`.
baseline = WalkForward(test_size=21, train_size=252, purged_size=1)
pred_wf = cross_val_predict(
    InverseVolatility(),
    X_test,
    cv=baseline,
    portfolio_params=dict(name="WalkForward"),
)

# %%
# Regime-filtered training
# ========================
# The same grid, but each training window is decoded with a two-state HMM on
# mean/volatility features. ``train_scope="current_regime"`` keeps the last
# contiguous run of the state observed at the end of the window.
cv = RegimeWalkForward(
    test_size=21,
    train_size=252,
    purged_size=1,
    detector=GaussianHMMDetector(
        n_regimes=2,
        feature="mean_vol",
        min_regime_size=21,
        random_state=0,
    ),
    train_scope="current_regime",
    min_train_size=63,
)
pred_regime = cross_val_predict(
    InverseVolatility(),
    X_test,
    cv=cv,
    portfolio_params=dict(name="RegimeWalkForward"),
)

print(f"WalkForward splits: {baseline.get_n_splits(X_test)}")
print(f"RegimeWalkForward splits: {cv.get_n_splits(X_test)}")
print(pred_wf.summary())
print(pred_regime.summary())

# %%
# Compare cumulative returns
# ==========================
population = Population([pred_wf, pred_regime])
fig = population.plot_cumulative_returns()
show(fig)

# %%
# Inspect one causal split
# ========================
train, test = next(cv.split(X_test))
assert train.max() < test.min()
print(f"Train size after regime filter: {len(train)}")
print(f"Test size: {len(test)}")
print(f"Train end / test start: {train.max()} / {test.min()}")
