"""Tests for portfolio calculations: optimisers, risk metrics, return transforms."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.calculations import (
    annualize,
    cagr,
    calmar_ratio,
    capm_regression,
    expected_shortfall,
    log_returns_from_prices,
    max_drawdown,
    max_sharpe_portfolio,
    min_volatility_portfolio,
    portfolio_returns,
    portfolio_summary,
    returns_from_prices,
    rolling_sharpe,
    rolling_volatility,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
    volatility,
    yearly_returns,
)


@pytest.fixture
def rng_returns() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(0.0005, 0.01, size=1000)


def test_returns_from_prices_basic():
    p = np.array([100.0, 110.0, 99.0])
    r = returns_from_prices(p)
    np.testing.assert_allclose(r, [0.10, -0.10], atol=1e-9)


def test_log_returns_consistency():
    p = np.array([100.0, 105.0, 110.0])
    r = returns_from_prices(p)
    lr = log_returns_from_prices(p)
    np.testing.assert_allclose(lr, np.log1p(r), atol=1e-9)


def test_annualize_scaling():
    mu_a, sd_a = annualize(0.001, 0.02)
    assert mu_a == pytest.approx(0.001 * 252)
    assert sd_a == pytest.approx(0.02 * np.sqrt(252))


def test_cagr_matches_compound_growth(rng_returns):
    c = cagr(rng_returns)
    expected = np.prod(1.0 + rng_returns) ** (252 / len(rng_returns)) - 1.0
    assert c == pytest.approx(expected, rel=1e-9)


def test_volatility_positive(rng_returns):
    assert volatility(rng_returns) > 0


def test_sharpe_zero_when_no_excess(rng_returns):
    """Sharpe of pure noise around rf should be roughly 0."""
    rf_daily = float(rng_returns.mean())
    excess = rng_returns - rf_daily
    s = sharpe_ratio(excess, rf=0.0)
    assert abs(s) < 0.5


def test_sortino_non_negative_for_positive_drift():
    """Sortino should be positive when mean excess return is positive."""
    rng = np.random.default_rng(3)
    r = rng.normal(0.001, 0.01, size=2000)
    assert sortino_ratio(r, rf=0.0) > 0


def test_sortino_finite(rng_returns):
    s = sortino_ratio(rng_returns, rf=0.0)
    assert np.isfinite(s)


def test_max_drawdown_non_positive():
    r = np.array([0.05, -0.10, 0.02, -0.05, 0.03])
    assert max_drawdown(r) <= 0


def test_var_cvar_ordering(rng_returns):
    var = value_at_risk(rng_returns, 0.05)
    cvar = expected_shortfall(rng_returns, 0.05)
    assert cvar >= var  # tail mean is worse than the quantile


def test_portfolio_returns_sums_to_one():
    r = np.random.default_rng(0).normal(0, 0.01, (500, 3))
    w = np.array([0.4, 0.4, 0.2])
    p = portfolio_returns(r, w)
    expected = r @ w
    np.testing.assert_allclose(p, expected, atol=1e-12)


def test_portfolio_summary_keys(rng_returns):
    r = np.column_stack([rng_returns, rng_returns * 0.5 + 0.0002])
    s = portfolio_summary(r, np.array([0.6, 0.4]))
    expected = {"annual_return", "annual_vol", "sharpe", "sortino", "calmar",
                "max_drawdown", "VaR_95", "CVaR_95", "VaR_99", "CVaR_99"}
    assert expected.issubset(s.keys())


def test_optimisers_sum_to_one():
    mu = np.array([0.10, 0.07, 0.12, 0.04])
    A = np.array([[0.04, 0.01, 0.02, 0.0],
                  [0.01, 0.05, 0.01, 0.0],
                  [0.02, 0.01, 0.07, 0.0],
                  [0.0, 0.0, 0.0, 0.02]])
    ms = max_sharpe_portfolio(mu, A, rf=0.02, long_only=True, max_weight=0.5)
    mv = min_volatility_portfolio(mu, A, long_only=True, max_weight=1.0)
    np.testing.assert_allclose(ms.sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(mv.sum(), 1.0, atol=1e-6)
    assert (ms >= -1e-9).all()
    assert (ms <= 0.5 + 1e-9).all()


def test_calmar_ratio_definition():
    r = np.array([0.05, -0.10, 0.02, -0.05, 0.03])
    c = calmar_ratio(r)
    expected_cagr = (np.prod(1 + r)) ** (252 / len(r)) - 1
    expected_mdd = abs(max_drawdown(r))
    assert c == pytest.approx(expected_cagr / expected_mdd, rel=1e-6)


# ---------------------------------------------------------------------------
# CAPM regression
# ---------------------------------------------------------------------------
def test_capm_regression_known_case():
    """If portfolio = benchmark exactly, alpha = 0, beta = 1, R^2 = 1."""
    rng = np.random.default_rng(11)
    bench = rng.normal(0.0005, 0.01, 500)
    out = capm_regression(bench.copy(), bench.copy(), rf=0.0)
    assert out["beta"] == pytest.approx(1.0, abs=1e-9)
    assert out["alpha_daily"] == pytest.approx(0.0, abs=1e-9)
    assert out["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_capm_regression_uncorrelated_low_r2():
    """Independent portfolio should have low R-squared."""
    rng = np.random.default_rng(12)
    port = rng.normal(0.0005, 0.01, 500)
    bench = rng.normal(0.0005, 0.01, 500)
    out = capm_regression(port, bench, rf=0.0)
    assert -1.0 < out["beta"] < 1.0
    assert out["r_squared"] < 0.1


def test_capm_alpha_annual_252x_daily():
    rng = np.random.default_rng(13)
    port = rng.normal(0.001, 0.01, 500)
    bench = rng.normal(0.0005, 0.01, 500)
    out = capm_regression(port, bench, rf=0.0)
    assert out["alpha_annual"] == pytest.approx(out["alpha_daily"] * 252, rel=1e-9)


# ---------------------------------------------------------------------------
# Rolling metrics
# ---------------------------------------------------------------------------
def test_rolling_sharpe_shape_and_first_nan():
    rng = np.random.default_rng(14)
    r = rng.normal(0.0005, 0.01, 500)
    rs = rolling_sharpe(r, window=60)
    assert len(rs) == 500
    assert bool(np.all(rs[:59] == 0))
    assert bool(np.all(np.isfinite(rs[59:])))


def test_rolling_sharpe_zero_vol_window():
    """Window with all-zero returns gives 0 (the guard), not NaN/Inf."""
    r = np.zeros(100)
    rs = rolling_sharpe(r, window=10)
    assert np.isfinite(rs).all()


def test_rolling_volatility_basic():
    rng = np.random.default_rng(15)
    r = rng.normal(0.0, 0.01, 200)
    rv = rolling_volatility(r, window=30)
    assert len(rv) == 200
    assert bool(np.all(rv[:29] == 0))
    assert bool(np.all(rv[29:] > 0))


# ---------------------------------------------------------------------------
# Yearly returns
# ---------------------------------------------------------------------------
def test_yearly_returns_groups_calendar_years():
    dates = pd.date_range("2021-06-01", "2024-12-31", freq="B")
    r = np.full(len(dates), 0.001)
    df = yearly_returns(r, dates)
    assert list(df.index) == [2021, 2022, 2023, 2024]
    assert df.loc[2021, "Return"] > 0