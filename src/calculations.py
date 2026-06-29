"""
calculations.py - Modern Portfolio Theory math.

All routines are dependency-light (numpy + scipy) and stateless so they're
easy to test. Functions take pre-computed returns and weights and return
metrics. Portfolio construction is delegated to ``optimizers.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Returns preprocessing
# ---------------------------------------------------------------------------
def annualize(mean_daily: float, std_daily: float, periods: int = 252) -> tuple[float, float]:
    """Convert daily mean/std to annual figures. Assumes ~252 trading days."""
    return mean_daily * periods, std_daily * np.sqrt(periods)


def returns_from_prices(prices: np.ndarray) -> np.ndarray:
    """Simple arithmetic returns, dropping the first NaN row."""
    p = np.asarray(prices, dtype=float)
    return p[1:] / p[:-1] - 1.0


def log_returns_from_prices(prices: np.ndarray) -> np.ndarray:
    """Log returns, useful for heavier-tailed simulation."""
    p = np.asarray(prices, dtype=float)
    return np.log(p[1:] / p[:-1])


# ---------------------------------------------------------------------------
# Risk metrics (portfolio-level)
# ---------------------------------------------------------------------------
def portfolio_returns(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Linear return aggregation: r_p_t = w^T r_t."""
    return returns @ np.asarray(weights, dtype=float)


def cagr(returns: np.ndarray, periods: int = 252) -> float:
    """Compound annual growth rate from a return series."""
    g = float(np.prod(1.0 + returns))
    n = len(returns)
    if n == 0 or g <= 0:
        return 0.0
    return g ** (periods / n) - 1.0


def volatility(returns: np.ndarray, periods: int = 252) -> float:
    """Annualised standard deviation of returns."""
    return float(np.std(returns, ddof=1) * np.sqrt(periods))


def sharpe_ratio(returns: np.ndarray, rf: float = 0.0, periods: int = 252) -> float:
    """Sharpe = (annualised mean excess return) / annualised std."""
    excess = np.asarray(returns) - rf / periods
    sd = np.std(excess, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(periods))


def sortino_ratio(returns: np.ndarray, rf: float = 0.0, periods: int = 252) -> float:
    """Sharpe-like ratio that penalises only downside volatility."""
    excess = np.asarray(returns) - rf / periods
    downside = excess[excess < 0]
    dd = np.std(downside, ddof=1) if len(downside) > 1 else 1e-12
    return float(np.mean(excess) / dd * np.sqrt(periods))


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction."""
    equity = np.cumprod(1.0 + np.asarray(returns))
    peaks = np.maximum.accumulate(equity)
    drawdown = equity / peaks - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: np.ndarray, periods: int = 252) -> float:
    """CAGR divided by |max drawdown|; useful complement to Sharpe."""
    mdd = abs(max_drawdown(returns))
    c = cagr(returns, periods=periods)
    return float(c / mdd) if mdd > 1e-9 else 0.0


def value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Historical VaR as a positive loss number (e.g. 0.02 = 2%)."""
    return float(-np.quantile(returns, alpha))


def expected_shortfall(returns: np.ndarray, alpha: float = 0.05) -> float:
    """CVaR / Expected Shortfall = mean loss beyond the VaR tail."""
    var = value_at_risk(returns, alpha)
    tail = returns[returns <= -var]
    return float(-tail.mean()) if len(tail) else var


def portfolio_summary(returns: np.ndarray, weights: np.ndarray, rf: float = 0.0) -> dict:
    """Compute every standard metric at once for display in tables."""
    port = portfolio_returns(returns, weights)
    return {
        "annual_return": cagr(port),
        "annual_vol": volatility(port),
        "sharpe": sharpe_ratio(port, rf=rf),
        "sortino": sortino_ratio(port, rf=rf),
        "calmar": calmar_ratio(port),
        "max_drawdown": max_drawdown(port),
        "VaR_95": value_at_risk(port, 0.05),
        "CVaR_95": expected_shortfall(port, 0.05),
        "VaR_99": value_at_risk(port, 0.01),
        "CVaR_99": expected_shortfall(port, 0.01),
    }


# ---------------------------------------------------------------------------
# Mean-Variance optimisation (Markowitz, 1952)
# ---------------------------------------------------------------------------
def _neg_sharpe(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> float:
    port = mu @ weights
    sd = np.sqrt(weights @ cov @ weights)
    return -((port - rf) / sd) if sd > 1e-12 else 1e6


def _port_vol(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov @ weights))


def _port_return(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    return float(mu @ weights)


def _weights_constraints(n: int, long_only: bool, max_w: float):
    bounds = [(0.0, max_w) if long_only else (-1.0, 1.0) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    return bounds, constraints


def max_sharpe_portfolio(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.0,
    long_only: bool = True,
    max_weight: float = 1.0,
) -> np.ndarray:
    """Tangency portfolio: maximises Sharpe under sum-to-one + optional long-only."""
    n = len(mu)
    x0 = np.ones(n) / n
    bounds, constraints = _weights_constraints(n, long_only, max_weight)
    res = minimize(_neg_sharpe, x0, args=(mu, cov, rf), method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"maxiter": 500, "ftol": 1e-9})
    w = res.x
    w = np.clip(w, 0, None) if long_only else w
    s = w.sum()
    return w / s if s > 1e-12 else x0


def min_volatility_portfolio(
    mu: np.ndarray,
    cov: np.ndarray,
    long_only: bool = True,
    max_weight: float = 1.0,
) -> np.ndarray:
    """Global minimum variance portfolio (Markowitz MVP)."""
    n = len(mu)
    x0 = np.ones(n) / n
    bounds, constraints = _weights_constraints(n, long_only, max_weight)
    res = minimize(_port_vol, x0, args=(mu, cov), method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"maxiter": 500, "ftol": 1e-9})
    w = np.clip(res.x, 0, None) if long_only else res.x
    s = w.sum()
    return w / s if s > 1e-12 else x0


def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    n_points: int = 50,
    long_only: bool = True,
    max_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Trace out the efficient frontier by sweeping target returns."""
    n = len(mu)
    bounds, _ = _weights_constraints(n, long_only, max_weight)
    ret_min, ret_max = float(mu.min()), float(mu.max())
    targets = np.linspace(ret_min, ret_max, n_points)
    rets, vols, weights_list = [], [], []
    for tgt in targets:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=tgt: mu @ w - t},
        ]
        x0 = np.ones(n) / n
        r = minimize(_port_vol, x0, args=(mu, cov), method="SLSQP",
                     bounds=bounds, constraints=cons,
                     options={"maxiter": 300, "ftol": 1e-9})
        if r.success:
            w = np.clip(r.x, 0, None) if long_only else r.x
            s = w.sum()
            if s > 1e-12:
                w = w / s
                rets.append(mu @ w)
                vols.append(np.sqrt(w @ cov @ w))
                weights_list.append(w)
    return np.array(rets), np.array(vols), weights_list


def random_portfolios(
    mu: np.ndarray, cov: np.ndarray, n: int = 2000, long_only: bool = True, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monte-Carlo sample of feasible portfolios to underlay the frontier plot."""
    rng = np.random.default_rng(seed)
    k = len(mu)
    rets, vols, weights = np.empty(n), np.empty(n), np.empty((n, k))
    for i in range(n):
        w = rng.random(k)
        w = w / w.sum()
        if long_only:
            w = np.clip(w, 0, None)
            w = w / w.sum()
        weights[i] = w
        rets[i] = mu @ w
        vols[i] = np.sqrt(w @ cov @ w)
    return rets, vols, weights


# ---------------------------------------------------------------------------
# CAPM single-factor regression: r_p = alpha + beta * r_b + epsilon
# ---------------------------------------------------------------------------
def capm_regression(port_ret: np.ndarray, bench_ret: np.ndarray, rf: float = 0.0) -> dict:
    """OLS fit of portfolio excess returns on benchmark excess returns.

    Returns alpha (annualised), beta, R-squared, residual std and the
    predicted (systematic) series.
    """
    pr = np.asarray(port_ret, dtype=float)
    br = np.asarray(bench_ret, dtype=float)
    n = min(len(pr), len(br))
    pr, br = pr[-n:], br[-n:]
    excess_p = pr - rf / 252
    excess_b = br - rf / 252

    X = np.column_stack([np.ones(n), excess_b])
    coef, *_ = np.linalg.lstsq(X, excess_p, rcond=None)
    alpha_d, beta = float(coef[0]), float(coef[1])

    fitted = alpha_d + beta * excess_b
    resid = excess_p - fitted
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((excess_p - excess_p.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {
        "alpha_daily": alpha_d,
        "alpha_annual": alpha_d * 252,
        "beta": beta,
        "r_squared": r2,
        "resid_std_daily": float(resid.std(ddof=1)),
        "tracking_error_annual": float(resid.std(ddof=1) * np.sqrt(252)),
        "n_obs": n,
    }


def rolling_sharpe(returns: np.ndarray, window: int = 60, periods: int = 252, rf: float = 0.0) -> np.ndarray:
    """Rolling annualised Sharpe ratio over a sliding window. Returns 0 where insufficient history."""
    r = np.asarray(returns, dtype=float)
    excess = r - rf / periods
    n = len(excess)
    out = np.zeros(n)
    for i in range(window - 1, n):
        window_slice = excess[i - window + 1: i + 1]
        sd = window_slice.std(ddof=1)
        if sd > 1e-12:
            out[i] = window_slice.mean() / sd * np.sqrt(periods)
    return out


def rolling_volatility(returns: np.ndarray, window: int = 60, periods: int = 252) -> np.ndarray:
    """Rolling annualised volatility. Returns 0 where insufficient history."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    out = np.zeros(n)
    for i in range(window - 1, n):
        out[i] = r[i - window + 1: i + 1].std(ddof=1) * np.sqrt(periods)
    return out


def yearly_returns(returns: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Annual total return per calendar year, plus a YoY summary table."""
    yearly = (1.0 + pd.Series(returns, index=index)).resample("YE").prod() - 1.0
    out = pd.DataFrame({"Year": yearly.index.year, "Return": yearly.values})
    return out.set_index("Year")