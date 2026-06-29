"""
data.py - Market data layer.

Loads adjusted close prices via yfinance, caches them in SQLite so repeated
runs in Streamlit don't hit the network, and falls back to a synthetic
correlated panel if yfinance isn't available or no tickers are supplied.
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Synthetic data (always available; used as a no-network fallback)
# ---------------------------------------------------------------------------
ASSETS = ["EQ_US", "EQ_EU", "EQ_EM", "BOND_US", "GOLD", "REIT"]


def make_synthetic(n_days: int = 756, seed: int = 42) -> pd.DataFrame:
    """A correlated synthetic price panel with realistic vol / drift."""
    rng = np.random.default_rng(seed)
    k = len(ASSETS)
    mu = np.array([0.08, 0.07, 0.10, 0.03, 0.05, 0.06]) / 252
    vol = np.array([0.18, 0.20, 0.28, 0.06, 0.16, 0.22]) / np.sqrt(252)
    L = np.linalg.cholesky(0.3 * np.ones((k, k)) + 0.7 * np.eye(k))
    R = mu + (rng.normal(size=(n_days, k)) @ L.T) * vol
    prices = 100.0 * np.cumprod(1.0 + R, axis=0)
    dates = pd.bdate_range(end=datetime.today().date(), periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=ASSETS)


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------
def _cache_path(db_dir: str | Path = "data/cache.db") -> Path:
    Path(db_dir).parent.mkdir(parents=True, exist_ok=True)
    return Path(db_dir)


def _cache_get(ticker: str, start: str, end: str, db_path: Path) -> pd.Series | None:
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(
                "SELECT date, close FROM prices WHERE ticker=? AND date BETWEEN ? AND ? ORDER BY date",
                conn, params=(ticker, start, end), parse_dates=["date"], index_col="date"
            )
        if df.empty or len(df) < 5:
            return None
        return df["close"]
    except Exception:
        return None


def _cache_put(ticker: str, close: pd.Series, db_path: Path) -> None:
    if close.empty:
        return
    df = pd.DataFrame({"date": close.index, "ticker": ticker, "close": close.values})
    with sqlite3.connect(db_path) as conn:
        df.to_sql("prices", conn, if_exists="append", index=False)


# ---------------------------------------------------------------------------
# yfinance loaders
# ---------------------------------------------------------------------------
def _to_business_dates(series: pd.Series) -> pd.Series:
    """yfinance occasionally returns tz-aware indexes; normalise to dates."""
    s = series.copy()
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s


def load_yfinance(
    tickers: Iterable[str],
    period: str = "3y",
    use_cache: bool = True,
    cache_db: str | Path = "data/cache.db",
    retries: int = 3,
    backoff: float = 1.5,
) -> pd.DataFrame:
    """Adjusted close prices as a wide DataFrame (date x ticker)."""
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers:
        raise ValueError("At least one ticker is required")
    db_path = _cache_path(cache_db)
    end_dt = datetime.today().date()
    start_dt = end_dt - _period_to_timedelta(period)
    start, end = start_dt.isoformat(), end_dt.isoformat()

    frames: dict[str, pd.Series] = {}
    for t in tickers:
        cached = _cache_get(t, start, end, db_path) if use_cache else None
        if cached is not None:
            frames[t] = cached
            continue
        frames[t] = _fetch_with_retries(t, retries, backoff, db_path, use_cache)

    df = pd.concat(frames.values(), axis=1, keys=frames.keys()).sort_index().ffill().dropna()
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df


def _fetch_with_retries(
    ticker: str, retries: int, backoff: float, db_path: Path, use_cache: bool
) -> pd.Series:
    import yfinance as yf  # imported lazily so synthetic mode works offline
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            data = yf.Ticker(ticker).history(period="max", auto_adjust=True)["Close"]
            data = _to_business_dates(data)
            if data.empty:
                raise RuntimeError(f"empty history for {ticker}")
            if use_cache:
                _cache_put(ticker, data, db_path)
            return data
        except Exception as exc:  # noqa: BLE001 - yfinance raises many subtypes
            last_err = exc
            time.sleep(backoff ** attempt)
    raise RuntimeError(f"Could not fetch {ticker} after {retries} attempts: {last_err}")


def _period_to_timedelta(period: str) -> timedelta:
    period = period.lower().strip()
    units = {"d": 1, "mo": 30, "y": 365}
    for suffix, mult in units.items():
        if period.endswith(suffix):
            return timedelta(days=int(period[: -len(suffix)]) * mult)
    return timedelta(days=int(period))


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
def prices_to_returns(prices: pd.DataFrame, method: str = "arithmetic") -> pd.DataFrame:
    """Convert a wide price panel to a return panel."""
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


def estimate_mu_cov(returns: pd.DataFrame, periods: int = 252) -> tuple[np.ndarray, np.ndarray]:
    """Annualised mean vector and covariance matrix."""
    mu = returns.mean().to_numpy() * periods
    cov = returns.cov().to_numpy() * periods
    return mu, cov