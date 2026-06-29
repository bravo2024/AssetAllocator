"""Efficient Frontier tab."""
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
    max_sharpe_portfolio, min_volatility_portfolio, efficient_frontier,
    random_portfolios, portfolio_returns, portfolio_summary,
)
from src.visualizations import (
    efficient_frontier_chart, weights_bar, backtest_compare_chart,
)

st.title("Efficient Frontier")

st.caption(
    "Markowitz mean-variance: the parabolic frontier is the set of (return, volatility) "
    "combinations you can't improve on without taking more risk. Yellow star = tangency "
    "(Max Sharpe), red diamond = minimum variance."
)

nav = st.columns(5)
nav[0].page_link("main.py", label="Home")
nav[1].page_link("pages/Efficient_Frontier.py", label="Efficient Frontier", disabled=True)
nav[2].page_link("pages/Risk_Dashboard.py", label="Risk Dashboard")
nav[3].page_link("pages/Performance.py", label="Performance")
nav[4].page_link("pages/Monte_Carlo.py", label="Monte Carlo")

mu = st.session_state["mu"]
cov = st.session_state["cov"]
returns: pd.DataFrame = st.session_state["returns"]
tickers = st.session_state["tickers"]
rf = st.session_state["rf"]
long_only = st.session_state["long_only"]
max_weight = st.session_state["max_weight"]
n_random = st.session_state["n_random"]

with st.spinner("Solving the optimisation problems..."):
    ms_w = max_sharpe_portfolio(mu, cov, rf=rf, long_only=long_only, max_weight=max_weight)
    mv_w = min_volatility_portfolio(mu, cov, long_only=long_only, max_weight=max_weight)
    f_rets, f_vols, _ = efficient_frontier(mu, cov, n_points=40, long_only=long_only, max_weight=max_weight)
    r_rets, r_vols, _ = random_portfolios(mu, cov, n=n_random, long_only=long_only)

asset_vols = np.sqrt(np.diag(cov))
asset_rets = mu
ms_summary = portfolio_summary(returns.to_numpy(), ms_w, rf=rf)
mv_summary = portfolio_summary(returns.to_numpy(), mv_w, rf=rf)
ew_w = np.ones(len(tickers)) / len(tickers)
ew_summary = portfolio_summary(returns.to_numpy(), ew_w, rf=rf)

fig = efficient_frontier_chart(
    f_rets, f_vols, r_rets, r_vols,
    float(ms_summary["annual_return"]), float(ms_summary["annual_vol"]),
    float(mv_summary["annual_return"]), float(mv_summary["annual_vol"]),
    asset_rets, asset_vols, tickers,
)
st.plotly_chart(fig, width='stretch')
st.caption(
    "Grey cloud: 1,500 randomly sampled long-only portfolios. Blue curve: efficient frontier. "
    "Black X marks: individual assets. Yellow star: tangency (best risk-adjusted return). "
    "Red diamond: minimum-variance portfolio."
)

st.markdown("### Optimal portfolio weights")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(weights_bar(ms_w, tickers, "Maximum Sharpe Ratio"), width='stretch')
with col2:
    st.plotly_chart(weights_bar(mv_w, tickers, "Minimum Volatility"), width='stretch')

with st.expander("Raw weight tables"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Max Sharpe**")
        st.dataframe(
            {"Ticker": tickers, "Weight": [f"{w:.2%}" for w in ms_w]},
            hide_index=True, width='stretch'
        )
    with col2:
        st.markdown("**Min Volatility**")
        st.dataframe(
            {"Ticker": tickers, "Weight": [f"{w:.2%}" for w in mv_w]},
            hide_index=True, width='stretch'
        )

st.markdown("### Historical backtest of the three strategies")
st.caption(
    "Same data window as the optimisation. All strategies start at $1 on the left. The "
    "tangency portfolio should dominate the equal-weight benchmark on a risk-adjusted basis, "
    "but the minimum-variance portfolio usually has the smoothest drawdown."
)
ms_ret = pd.Series(portfolio_returns(returns.to_numpy(), ms_w), index=returns.index)
mv_ret = pd.Series(portfolio_returns(returns.to_numpy(), mv_w), index=returns.index)
ew_ret = pd.Series(portfolio_returns(returns.to_numpy(), ew_w), index=returns.index)
st.plotly_chart(
    backtest_compare_chart(
        {"Max Sharpe": ms_ret, "Min Volatility": mv_ret, "Equal weight": ew_ret},
        title="$1 \u2192 ? over the panel",
    ),
    width='stretch',
)

st.markdown("### Side-by-side metrics")
col1, col2, col3, col4 = st.columns(4)
metrics = [
    ("Max Sharpe", ms_summary, col1),
    ("Min Volatility", mv_summary, col2),
    ("Equal weight", ew_summary, col3),
]
with col4:
    st.markdown("**Notes**")
    st.caption(
        "CAGR = compound annual growth rate. Vol = annualised std-dev of daily returns. "
        "Sharpe = excess return / vol. MDD = worst peak-to-trough drop."
    )
for label, s, col in metrics:
    with col:
        st.markdown(f"**{label}**")
        st.metric("CAGR", f"{s['annual_return']:.2%}",
                  help="Compound annual growth rate.")
        st.metric("Vol", f"{s['annual_vol']:.2%}",
                  help="Annualised std-dev of daily returns.")
        st.metric("Sharpe", f"{s['sharpe']:.2f}",
                  help="Excess return per unit of volatility.")
        st.metric("Max DD", f"{s['max_drawdown']:.2%}",
                  help="Worst peak-to-trough drawdown over the panel.")