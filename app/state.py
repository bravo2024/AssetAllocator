"""state.py - shared session-state bootstrap for app pages.

Streamlit pages only see ``session_state`` populated if the user lands on the
home page first (which is what runs ``app/main.py``). Direct navigation to
a sub-page would otherwise raise ``KeyError``. This helper loads the data
on demand using the same defaults as the home page sidebar so any tab works
in isolation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import make_synthetic, load_yfinance, prices_to_returns, estimate_mu_cov  # noqa: E402

DEFAULT_TICKERS = ["SPY", "EFA", "EEM", "AGG", "GLD", "VNQ"]


@st.cache_data(show_spinner="Downloading market data...")
def _fetch_prices(tickers_str: str, period: str, source: str) -> tuple[pd.DataFrame, str]:
    if source == "Synthetic" or not tickers_str.strip():
        return make_synthetic(), "synthetic"
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    try:
        df = load_yfinance(tickers, period=period, use_cache=True)
        return df, "live"
    except Exception:  # noqa: BLE001
        return make_synthetic(), "synthetic"


def ensure_state(
    default_tickers: list[str] | None = None,
    default_period: str = "3y",
    default_rf: float = 0.04,
    default_long_only: bool = True,
    default_max_weight: float = 0.40,
    default_n_random: int = 1500,
) -> None:
    """Always render the sidebar + fetch data if not already in session_state.

    ``_fetch_prices`` is ``@st.cache_data``-decorated, so repeated calls with
    unchanged inputs are instant cache hits.
    """
    already_loaded = "mu" in st.session_state

    if default_tickers is None:
        default_tickers = DEFAULT_TICKERS

    # Allow the user to override via sidebar even on direct page loads.
    with st.sidebar:
        st.header("Configuration")
        data_source = st.radio(
            "Data source", ["Live (yfinance)", "Synthetic"], index=0,
            help="Live pulls adjusted close prices from Yahoo Finance. Synthetic is an offline correlated panel."
        )
        tickers_input = st.text_input(
            "Tickers (comma-separated)", value=",".join(default_tickers),
            help="Any Yahoo Finance symbol.",
        )
        period = st.selectbox(
            "Lookback period", ["1y", "2y", "3y", "5y", "10y"],
            index=["1y", "2y", "3y", "5y", "10y"].index(default_period),
        )
        rf = st.number_input(
            "Risk-free rate (annualised)", value=default_rf, min_value=0.0, max_value=0.20,
            step=0.005, format="%.3f",
        )
        long_only = st.checkbox("Long-only weights", value=default_long_only)
        max_weight = st.slider(
            "Max weight per asset", 0.05, 1.0, default_max_weight, 0.05,
        )
        n_random = st.slider(
            "Random portfolios", 200, 5000, default_n_random, 100,
        )

    if not already_loaded:
        prices, source_used = _fetch_prices(tickers_input, period, data_source)
        returns = prices_to_returns(prices, method="arithmetic")
        mu, cov = estimate_mu_cov(returns)

        st.session_state["prices"] = prices
        st.session_state["returns"] = returns
        st.session_state["mu"] = mu
        st.session_state["cov"] = cov
        st.session_state["rf"] = float(rf)
        st.session_state["long_only"] = bool(long_only)
        st.session_state["max_weight"] = float(max_weight)
        st.session_state["n_random"] = int(n_random)
        st.session_state["tickers"] = list(prices.columns)
        st.session_state["data_source"] = source_used