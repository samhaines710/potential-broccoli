import pandas as pd
import numpy as np
import pytest

from portfolio_optimization import (
    CVXMeanVarianceOptimizer,
    PyPortfolioOptOptimizer,
    RiskParityOptimizer
)

def random_returns(n_assets=4, n_periods=100):
    np.random.seed(0)
    data = np.random.normal(0,0.01,size=(n_periods,n_assets))
    idx = pd.date_range("2021-01-01", periods=n_periods)
    cols= [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(data, index=idx, columns=cols)

@pytest.fixture
def returns_df():
    return random_returns()

def test_cvx_optimizer(returns_df):
    opt = CVXMeanVarianceOptimizer(returns_df)
    w = opt.optimize(target_return=0.0)
    assert pytest.approx(w.sum(), rel=1e-3) == 1.0

def test_pyportfolioopt_optimizer(returns_df):
    opt = PyPortfolioOptOptimizer(returns_df)
    w1 = opt.max_sharpe()
    w2 = opt.min_volatility()
    assert pytest.approx(w1.sum(), abs=1e-6) == 1.0
    assert pytest.approx(w2.sum(), abs=1e-6) == 1.0

def test_risk_parity(returns_df):
    opt = RiskParityOptimizer(returns_df)
    w = opt.optimize()
    assert pytest.approx(w.sum(), abs=1e-6) == 1.0
