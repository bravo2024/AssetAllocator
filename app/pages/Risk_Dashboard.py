"""Risk Dashboard tab: historical VaR/CVaR + drawdown + correlation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from app.state import ensure_state
ensure_state()

from src.calculations import (
    max_sharpe_portfolio, portfolio_returns, value_at_risk, expected_shortfall,
)
from src.visualizations import (
    correlation_heatmap, drawdown_chart, return_distribution, cumulative_returns_chart,
    weight_donut,
)

st.title("Risk Dashboard")

st.caption(
    "Historical risk for any portfolio you can construct. Edit weights in the table on the "
    "left; metrics, donut, and charts update live."
)

nav = st.columns(5)
nav[0].page_link("main.py", label="Home")
nav[1].page_link("pages/Efficient_Frontier.py", label="Efficient Frontier")
nav[2].page_link("pages/Risk_Dashboard.py", label="Risk Dashboard", disabled=True)
nav[3].page_link("pages/Performance.py", label="Performance")
nav[4].page_link("pages/Monte_Carlo.py", label="Monte Carlo")

returns: pd.DataFrame = st.session_state["returns"]
mu: np.ndarray = st.session_state["mu"]
cov: np.ndarray = st.session_state["cov"]
rf: float = st.session_state["rf"]
long_only: bool = st.session_state["long_only"]
max_weight: float = st.session_state["max_weight"]

ms_w = max_sharpe_portfolio(mu, cov, rf=rf, long_only=long_only, max_weight=max_weight)
tickers = list(returns.columns)

# ---------------------------------------------------------------------------
# Left column: editable weight table. Right column: live donut + metrics.
# ---------------------------------------------------------------------------
left, right = st.columns([2, 3])

with left:
    st.markdown("### Weights")
    if "rd_weights" not in st.session_state:
        st.session_state.rd_weights = {t: float(w) for t, w in zip(tickers, ms_w)}

    presets = st.radio(
        "Preset", ["Max Sharpe", "Equal weight"], horizontal=True,
        key="rd_preset",
    )
    if presets == "Max Sharpe":
        st.session_state.rd_weights = {t: float(w) for t, w in zip(tickers, ms_w)}
    elif presets == "Equal weight":
        eq = 1.0 / len(tickers)
        st.session_state.rd_weights = {t: eq for t in tickers}

    edited = st.data_editor(
        pd.DataFrame(
            {"Ticker": tickers, "Weight": [st.session_state.rd_weights[t] for t in tickers]}
        ),
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
            "Weight": st.column_config.NumberColumn(
                "Weight", min_value=0.0, max_value=max_weight, step=0.01,
                format="%.2f",
            ),
        },
        hide_index=True, width='stretch', key="rd_editor",
    )
    weights = np.array(edited["Weight"].tolist(), dtype=float)
    if weights.sum() == 0:
        st.info("All weights zero; defaulting to Max Sharpe.")
        weights = ms_w
        weights = weights / weights.sum()
    else:
        weights = weights / weights.sum()

    st.caption(f"Sum normalised to 1.00 ({weights.sum():.2%} of capital deployed).")

with right:
    st.markdown("### Current allocation")
    st.plotly_chart(weight_donut(weights, tickers), width='stretch')

# ---------------------------------------------------------------------------
# Recompute metrics from edited weights
# ---------------------------------------------------------------------------
port_ret = pd.Series(portfolio_returns(returns.to_numpy(), weights), index=returns.index)
var95 = value_at_risk(port_ret.to_numpy(), 0.05)
cvar95 = expected_shortfall(port_ret.to_numpy(), 0.05)
var99 = value_at_risk(port_ret.to_numpy(), 0.01)
cvar99 = expected_shortfall(port_ret.to_numpy(), 0.01)

st.markdown("### Portfolio risk (daily)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("VaR 95%", f"{var95:.2%}",
            help="Loss not exceeded on 95% of trading days (positive number).")
col2.metric("CVaR 95%", f"{cvar95:.2%}",
            help="Average loss on the worst 5% of days. Always \u2265 VaR. Basel II/III use this.")
col3.metric("VaR 99%", f"{var99:.2%}",
            help="Loss not exceeded on 99% of trading days. Stress-scenario measure.")
col4.metric("CVaR 99%", f"{cvar99:.2%}",
            help="Average loss on the worst 1% of days.")

col1, col2, col3 = st.columns(3)
col1.metric("Annualised vol", f"{port_ret.std() * np.sqrt(252):.2%}",
            help="Standard deviation of daily returns, scaled to a year.")
col2.metric("Skewness", f"{float(port_ret.skew()):.2f}",
            help="Negative = losses more extreme than gains.")
col3.metric("Excess kurtosis", f"{float(port_ret.kurtosis()):.2f}",
            help="Tail-heaviness vs a normal distribution. > 0 = fat tails.")

st.caption(
    "**Reference for context.** Normal distribution: skew = 0, excess kurtosis = 0. "
    "Broad equity indices (e.g. S&P 500) typically show skew \u2248 \u22120.3 and excess "
    "kurtosis \u2248 5\u201310 over multi-year windows. If your portfolio's skew is strongly "
    "negative *and* kurtosis > 5, the normal-GBM assumption understates tail risk."
)

# ---------------------------------------------------------------------------
# Charts in two columns to reduce vertical scroll
# ---------------------------------------------------------------------------
st.markdown("### Return distribution")
st.plotly_chart(return_distribution(port_ret, var95, cvar95), width='stretch')

st.markdown("### Drawdown and cumulative growth")
chart_left, chart_right = st.columns(2)
with chart_left:
    st.plotly_chart(drawdown_chart(port_ret), width='stretch')
with chart_right:
    st.plotly_chart(cumulative_returns_chart(port_ret), width='stretch')

st.markdown("### Pairwise return correlation")
st.plotly_chart(correlation_heatmap(returns), width='stretch')