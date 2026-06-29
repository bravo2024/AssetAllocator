from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data import make_synthetic, prices_to_returns, estimate_mu_cov
from src.calculations import max_sharpe_portfolio, min_volatility_portfolio


def test_data():
    prices = make_synthetic(100)
    assert prices.shape == (100, 6)


def test_returns():
    prices = make_synthetic(100)
    returns = prices_to_returns(prices)
    assert len(returns) == 99


def test_optimisers():
    prices = make_synthetic(300)
    returns = prices_to_returns(prices)
    mu, cov = estimate_mu_cov(returns)
    w = max_sharpe_portfolio(mu, cov, rf=0.04)
    assert abs(w.sum() - 1.0) < 1e-6


def test_min_vol():
    prices = make_synthetic(300)
    returns = prices_to_returns(prices)
    mu, cov = estimate_mu_cov(returns)
    w = min_volatility_portfolio(mu, cov)
    assert abs(w.sum() - 1.0) < 1e-6
