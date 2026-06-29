"""Monte Carlo tab: simulate correlated GBM paths."""
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
from src.simulation import gbm_paths, simulate_var
from src.visualizations import monte_carlo_paths, monte_carlo_terminal_hist

st.title("Monte Carlo Simulation")

st.caption(
    "Forward-looking risk: simulate thousands of correlated Geometric Brownian Motion "
    "paths and read off the probable loss distribution."
)

nav = st.columns(5)
nav[0].page_link("main.py", label="Home")
nav[1].page_link("pages/Efficient_Frontier.py", label="Efficient Frontier")
nav[2].page_link("pages/Risk_Dashboard.py", label="Risk Dashboard")
nav[3].page_link("pages/Performance.py", label="Performance")
nav[4].page_link("pages/Monte_Carlo.py", label="Monte Carlo", disabled=True)

mu = st.session_state["mu"]
cov = st.session_state["cov"]
returns: pd.DataFrame = st.session_state["returns"]
rf = st.session_state["rf"]
long_only = st.session_state["long_only"]
max_weight = st.session_state["max_weight"]
ms_w = max_sharpe_portfolio(mu, cov, rf=rf, long_only=long_only, max_weight=max_weight)

col1, col2, col3, col4 = st.columns(4)
horizon = col1.slider("Horizon (days)", 21, 504, 252, step=21,
                      help="Forward projection window. 252 = 1 year.")
n_sims = col2.slider("Simulations", 500, 20000, 5000, step=500,
                     help="Independent paths to draw. 5,000 takes ~1 second.")
initial = col3.number_input("Initial value ($)", value=10_000.0, step=1_000.0, format="%.0f",
                            help="Starting capital. P&L scales linearly with this.")
alpha = col4.selectbox("Confidence level", [0.01, 0.05, 0.10], index=1,
                       format_func=lambda x: f"{int((1 - x) * 100)}%",
                       help="VaR / CVaR reported at this confidence. 5% (95% confidence) is standard.")

tickers = st.session_state["tickers"]
mode = st.radio("Portfolio weights", ["Max Sharpe", "Equal weight", "Custom"], horizontal=True)
if mode == "Max Sharpe":
    w = ms_w
elif mode == "Equal weight":
    w = np.ones(len(tickers)) / len(tickers)
else:
    cols = st.columns(min(len(tickers), 6))
    w = np.array([cols[i % len(cols)].slider(t, 0.0, max_weight,
                                             float(v), step=0.01, key=f"mc_{t}")
                  for i, (t, v) in enumerate(zip(tickers, ms_w))])
    if w.sum() == 0:
        w = ms_w
    else:
        w = w / w.sum()

st.caption(
    "Method: correlated GBM with daily log-normal shocks via Cholesky factorisation of the "
    "annualised covariance matrix. Seed fixed at 42 for reproducibility."
)

with st.spinner(f"Running {n_sims:,} simulations..."):
    paths = gbm_paths(mu, cov, w, horizon_days=horizon, n_sims=n_sims, initial_value=initial, seed=42)
    var_stats = simulate_var(mu, cov, w, horizon_days=horizon, n_sims=n_sims,
                             alpha=alpha, initial_value=initial, seed=42)

# ---------------------------------------------------------------------------
# Headline metrics + probability of loss
# ---------------------------------------------------------------------------
st.markdown(f"### Simulated outcomes over {horizon} days at ${initial:,.0f} initial capital")
col1, col2, col3, col4 = st.columns(4)
col1.metric(f"VaR ({int((1 - alpha) * 100)}%)", f"${var_stats['VaR']:,.0f}",
            help=f"Worst loss not exceeded on {(1 - alpha) * 100:.0f}% of simulated paths.")
col2.metric(f"CVaR ({int((1 - alpha) * 100)}%)", f"${var_stats['CVaR']:,.0f}",
            help=f"Average loss on the worst {(1 - alpha) * 100:.0f}% of simulated paths.")
col3.metric("Mean P&L", f"${var_stats['mean']:,.0f}",
            help="Expected profit/loss across all simulations.")
col4.metric("P(loss)", f"{(var_stats['pnl'] < 0).mean():.1%}",
            help="Fraction of simulated paths ending with a loss. The single number a PM cares about most.")

# ---------------------------------------------------------------------------
# Historical vs simulated VaR comparison
# ---------------------------------------------------------------------------
st.markdown("### Historical vs simulated risk")
hist_port_ret = portfolio_returns(returns.to_numpy(), w)
hist_var = value_at_risk(hist_port_ret, alpha) * initial
hist_cvar = expected_shortfall(hist_port_ret, alpha) * initial
sim_var = var_stats["VaR"]
sim_cvar = var_stats["CVaR"]

comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
comp_col1.metric(f"Historical VaR ({int((1 - alpha) * 100)}%)", f"${hist_var:,.0f}",
                 help="Scaled to your initial capital from daily historical returns.")
comp_col2.metric(f"Simulated VaR ({int((1 - alpha) * 100)}%)", f"${sim_var:,.0f}",
                 help=f"Same confidence, from the {n_sims:,} simulated {horizon}-day paths.")
comp_col3.metric(f"Historical CVaR ({int((1 - alpha) * 100)}%)", f"${hist_cvar:,.0f}")
comp_col4.metric(f"Simulated CVaR ({int((1 - alpha) * 100)}%)", f"${sim_cvar:,.0f}")

var_diff = (sim_var - hist_var) / max(hist_var, 1.0)
cvar_diff = (sim_cvar - hist_cvar) / max(hist_cvar, 1.0)
if abs(var_diff) < 0.20:
    diff_msg = f"Historical and simulated VaR are within {abs(var_diff):.0%} of each other \u2014 GBM assumption holds."
elif var_diff > 0:
    diff_msg = (f"Simulated VaR is {var_diff:.0%} *higher* than historical. The forward-looking risk "
                f"is worse than the past would suggest \u2014 worth investigating.")
else:
    diff_msg = (f"Simulated VaR is {abs(var_diff):.0%} *lower* than historical. The portfolio may have "
                f"performed worse in the back-test than the normal-GBM forward model expects.")
st.caption(diff_msg)

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
st.plotly_chart(monte_carlo_paths(paths, n_show=120, n_highlight=5), width='stretch')
st.plotly_chart(monte_carlo_terminal_hist(var_stats["terminal_values"], initial), width='stretch')

st.markdown("### Terminal P&L quantiles")
quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
q_values = np.quantile(var_stats["pnl"], quantiles)
q_table = pd.DataFrame({
    "Quantile": [f"{q:.0%}" for q in quantiles],
    "P&L ($)": [f"${v:,.0f}" for v in q_values],
    "Interpretation": [
        "Tail loss (1-in-100 worst)", "VaR threshold", "1-in-10 worst", "Lower quartile",
        "Median outcome", "Upper quartile", "1-in-10 best", "1-in-20 best", "Tail gain (1-in-100 best)"
    ],
})
st.table(q_table)