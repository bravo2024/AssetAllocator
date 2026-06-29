"""Performance tab: cumulative, rolling Sharpe, CAPM, yearly returns, vs benchmark."""
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
    max_sharpe_portfolio, portfolio_returns, portfolio_summary,
    capm_regression, rolling_sharpe, rolling_volatility, yearly_returns,
)
from src.data import load_yfinance
from src.visualizations import (
    cumulative_returns_chart, monthly_heatmap, rolling_metric_chart,
)

st.title("Performance & Benchmark")

st.caption(
    "Compare your portfolio against a benchmark across cumulative return, rolling risk, "
    "monthly/yearly returns, and a CAPM single-factor regression."
)

nav = st.columns(5)
nav[0].page_link("main.py", label="Home")
nav[1].page_link("pages/Efficient_Frontier.py", label="Efficient Frontier")
nav[2].page_link("pages/Risk_Dashboard.py", label="Risk Dashboard")
nav[3].page_link("pages/Performance.py", label="Performance", disabled=True)
nav[4].page_link("pages/Monte_Carlo.py", label="Monte Carlo")

returns: pd.DataFrame = st.session_state["returns"]
mu: np.ndarray = st.session_state["mu"]
cov: np.ndarray = st.session_state["cov"]
rf: float = st.session_state["rf"]
long_only: bool = st.session_state["long_only"]
max_weight: float = st.session_state["max_weight"]
tickers: list[str] = st.session_state["tickers"]

benchmark = st.selectbox(
    "Benchmark", ["(none)", "SPY", "^GSPC", "QQQ", "IWM", "DIA"], index=1,
    help="Passive index for comparison. SPY = S&P 500, QQQ = Nasdaq 100, IWM = Russell 2000, DIA = Dow 30."
)
equal_weight = np.ones(len(tickers)) / len(tickers)
ms_w = max_sharpe_portfolio(mu, cov, rf=rf, long_only=long_only, max_weight=max_weight)

mode = st.radio("Portfolio", ["Max Sharpe", "Equal weight", "Custom"], horizontal=True)
if mode == "Max Sharpe":
    w = ms_w
elif mode == "Equal weight":
    w = equal_weight
else:
    cols = st.columns(min(len(tickers), 6))
    w = np.array([cols[i % len(cols)].slider(t, 0.0, max_weight,
                                             float(v), step=0.01, key=f"perf_{t}")
                  for i, (t, v) in enumerate(zip(tickers, ms_w))])
    if w.sum() == 0:
        w = ms_w
    else:
        w = w / w.sum()

port_ret = pd.Series(portfolio_returns(returns.to_numpy(), w), index=returns.index)
summary = portfolio_summary(returns.to_numpy(), w, rf=rf)

# Benchmark
bench_ret = None
if benchmark != "(none)":
    try:
        with st.spinner("Loading benchmark..."):
            bench_prices = load_yfinance([benchmark], period="5y", use_cache=True)
        aligned = bench_prices.reindex(returns.index).ffill()
        bench_ret = aligned.iloc[:, 0].pct_change().reindex(returns.index).fillna(0.0)
    except Exception:
        bench_ret = None

# ---------------------------------------------------------------------------
# Cumulative vs benchmark
# ---------------------------------------------------------------------------
st.markdown("### Cumulative growth vs benchmark")
if bench_ret is not None:
    st.caption(f"Solid line: your portfolio. Dashed line: {benchmark} over the same window.")
else:
    st.caption("Solid line: your portfolio. Select a benchmark above for a comparison line.")
st.plotly_chart(cumulative_returns_chart(port_ret, bench_ret), width='stretch')

# ---------------------------------------------------------------------------
# Rolling Sharpe + volatility
# ---------------------------------------------------------------------------
st.markdown("### Rolling 60-day Sharpe & volatility")
st.caption(
    "Annualised Sharpe and volatility computed on a trailing 60-trading-day window. "
    "Watch for regime shifts: sustained periods of negative rolling Sharpe often coincide "
    "with drawdowns in the chart above."
)
roll_sharpe = rolling_sharpe(port_ret.to_numpy(), window=60, rf=rf)
roll_vol = rolling_volatility(port_ret.to_numpy(), window=60)
series = {"Rolling Sharpe (60d)": roll_sharpe}
if bench_ret is not None:
    series["Rolling vol (60d, %)"] = roll_vol * 100
    series["Rolling Sharpe benchmark (60d)"] = rolling_sharpe(bench_ret.to_numpy(), window=60, rf=rf)
else:
    series["Rolling vol (60d, %)"] = roll_vol * 100

st.plotly_chart(
    rolling_metric_chart(series, returns.index, title="", ylabel="Sharpe (unitless) / Vol (%)"),
    width='stretch',
)

# ---------------------------------------------------------------------------
# Monthly heatmap + yearly returns table
# ---------------------------------------------------------------------------
st.markdown("### Returns calendar")
cal_left, cal_right = st.columns([3, 2])
with cal_left:
    st.markdown("**Monthly heatmap**")
    st.plotly_chart(monthly_heatmap(port_ret), width='stretch')
with cal_right:
    st.markdown("**Yearly returns**")
    yearly_df = yearly_returns(port_ret.to_numpy(), returns.index)
    yearly_df["Return"] = yearly_df["Return"].map(lambda v: f"{v:.2%}")
    st.table(yearly_df)

# ---------------------------------------------------------------------------
# CAPM attribution
# ---------------------------------------------------------------------------
st.markdown("### CAPM single-factor attribution")
if bench_ret is not None:
    capm = capm_regression(port_ret.to_numpy(), bench_ret.to_numpy(), rf=rf)
    cap_left, cap_right = st.columns([2, 3])
    with cap_left:
        st.metric("Alpha (annualised)", f"{capm['alpha_annual']:.2%}",
                  help="Jensen's alpha. Positive = outperformance vs benchmark, risk-adjusted.")
        st.metric("Beta", f"{capm['beta']:.2f}",
                  help="Sensitivity to benchmark moves. 1.0 = moves with the market, >1 = amplifies.")
        st.metric("R\u00b2", f"{capm['r_squared']:.2%}",
                  help="Fraction of portfolio variance explained by the benchmark. > 80% = market-driven.")
        st.metric("Tracking error (annualised)", f"{capm['tracking_error_annual']:.2%}",
                  help="Annualised std-dev of (portfolio \u2212 benchmark) returns.")
    with cap_right:
        capm_rows = [
            ("Alpha (daily)", f"{capm['alpha_daily']:.4%}"),
            ("Alpha (annualised)", f"{capm['alpha_annual']:.2%}"),
            ("Beta", f"{capm['beta']:.3f}"),
            ("R-squared", f"{capm['r_squared']:.2%}"),
            ("Tracking error (annualised)", f"{capm['tracking_error_annual']:.2%}"),
            ("Observations", f"{capm['n_obs']}"),
        ]
        st.table(pd.DataFrame(capm_rows, columns=["Quantity", "Value"]).set_index("Quantity"))
else:
    st.info("Pick a benchmark above to see alpha, beta and R\u00b2.")

# ---------------------------------------------------------------------------
# Full metric table
# ---------------------------------------------------------------------------
st.markdown("### Full metric table")
rows = [
    ("Annual return (CAGR)", f"{summary['annual_return']:.2%}", "Compound annual growth rate."),
    ("Annualised volatility", f"{summary['annual_vol']:.2%}", "Std dev of daily returns, scaled to a year."),
    ("Sharpe ratio", f"{summary['sharpe']:.2f}", "Excess return / total volatility."),
    ("Sortino ratio", f"{summary['sortino']:.2f}", "Excess return / downside-only volatility."),
    ("Calmar ratio", f"{summary['calmar']:.2f}", "CAGR / |max drawdown|."),
    ("Max drawdown", f"{summary['max_drawdown']:.2%}", "Worst peak-to-trough loss in the window."),
    ("VaR 95% (daily)", f"{summary['VaR_95']:.2%}", "1-in-20-day worst loss."),
    ("CVaR 95% (daily)", f"{summary['CVaR_95']:.2%}", "Average loss on the worst 5% of days."),
    ("VaR 99% (daily)", f"{summary['VaR_99']:.2%}", "1-in-100-day worst loss."),
    ("CVaR 99% (daily)", f"{summary['CVaR_99']:.2%}", "Average loss on the worst 1% of days."),
]
st.table(pd.DataFrame(rows, columns=["Metric", "Value", "Note"]).set_index("Metric"))