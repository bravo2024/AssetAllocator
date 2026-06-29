"""AssetAllocator entrypoint.

The ``streamlit run app/main.py`` command launches this file. Sidebar
controls live here; individual tabs are implemented in ``app/pages/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.state import ensure_state  # noqa: E402

st.set_page_config(
    page_title="AssetAllocator",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Project overview
# ---------------------------------------------------------------------------
st.title("AssetAllocator")
st.caption("Mean-variance portfolio optimisation on live market data with risk and Monte Carlo analytics.")

st.markdown(
    "A reference implementation of the Markowitz (1952) mean-variance framework, with the "
    "institutional risk stack (historical VaR / CVaR / Sharpe / Sortino / Calmar / drawdown) "
    "and a forward-looking Monte Carlo engine on correlated Geometric Brownian Motion."
)
st.caption(
    "**Three pillars:** \u00b7 portfolio construction under weight constraints \u00b7 "
    "historical risk decomposition \u00b7 simulation of probable loss distributions."
)

st.markdown("### Where to go next")
nav1, nav2 = st.columns(2)
with nav1:
    st.info(
        "**Efficient Frontier**\n\n"
        "Build the Markowitz frontier, see the random-portfolio cloud, compare "
        "Max Sharpe vs Min Volatility weights and backtest them side-by-side."
    )
    st.info(
        "**Risk Dashboard**\n\n"
        "Set any portfolio via an editable table + live donut, then read off "
        "VaR / CVaR at 95% & 99%, drawdown, return distribution and correlation."
    )
with nav2:
    st.info(
        "**Performance & Benchmark**\n\n"
        "Cumulative growth vs benchmark, monthly and yearly return tables, "
        "rolling 60-day Sharpe, CAPM alpha/beta/R\u00b2 attribution."
    )
    st.info(
        "**Monte Carlo**\n\n"
        "Run thousands of correlated GBM paths over any horizon; read off "
        "simulated VaR, probability of loss, and the full P&L distribution."
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Boot state (sidebar lives inside ensure_state)
# ---------------------------------------------------------------------------
ensure_state()

prices = st.session_state["prices"]
returns = st.session_state["returns"]
source_used = st.session_state.get("data_source", "live")

# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------
col_a, col_b, col_c, col_d, col_e = st.columns(5)
col_a.metric("Assets", len(prices.columns), help="Number of distinct tickers in the price panel.")
col_b.metric("Trading days", len(prices), help="Business days covered (~252 per year).")
col_c.metric("Start", str(prices.index.min().date()), help="Earliest observation.")
col_d.metric("End", str(prices.index.max().date()), help="Most recent observation.")
col_e.metric(
    "Source", source_used.upper(),
    help="LIVE = Yahoo Finance adjusted-close prices. SYNTHETIC = offline correlated panel."
)

# ---------------------------------------------------------------------------
# Price history + correlation side by side
# ---------------------------------------------------------------------------
left, right = st.columns([3, 2])
with left:
    st.markdown("#### Price history (rebased to 100)")
    from src.visualizations import normalised_price_chart  # noqa: E402
    st.plotly_chart(normalised_price_chart(prices), width='stretch')
with right:
    st.markdown("#### Pairwise return correlation")
    from src.visualizations import correlation_heatmap  # noqa: E402
    st.plotly_chart(correlation_heatmap(returns), width='stretch')
    st.caption(
        "Close to +1 = move together (low diversification). "
        "Close to \u22121 = hedge each other (high diversification)."
    )