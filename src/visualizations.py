"""
visualizations.py - Plotly figure factories used across the Streamlit app.

Each function returns a ``plotly.graph_objects.Figure`` so the Streamlit
layer can drop them straight into ``st.plotly_chart``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Price & return charts
# ---------------------------------------------------------------------------
def price_chart(prices: pd.DataFrame, title: str = "Adjusted Close") -> go.Figure:
    fig = px.line(prices, x=prices.index, y=prices.columns, title=title)
    fig.update_layout(legend_title_text="Ticker", hovermode="x unified",
                      height=420, template="plotly_white")
    return fig


def normalised_price_chart(prices: pd.DataFrame, title: str = "Normalised to 100") -> go.Figure:
    norm = prices / prices.iloc[0] * 100.0
    return price_chart(norm, title=title)


def correlation_heatmap(returns: pd.DataFrame) -> go.Figure:
    corr = returns.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title="Return correlation",
                    aspect="equal", origin="lower")
    fig.update_layout(height=460, template="plotly_white")
    return fig


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------
def efficient_frontier_chart(
    frontier_rets: np.ndarray,
    frontier_vols: np.ndarray,
    random_rets: np.ndarray,
    random_vols: np.ndarray,
    ms_sharpe_ret: float, ms_sharpe_vol: float,
    mv_ret: float, mv_vol: float,
    asset_rets: np.ndarray, asset_vols: np.ndarray, asset_labels: list[str],
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=random_vols, y=random_rets, mode="markers",
        marker=dict(size=5, color="rgba(120,120,120,0.35)"),
        name="Random portfolios", hovertemplate="σ=%{x:.2%}<br>μ=%{y:.2%}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=frontier_vols, y=frontier_rets, mode="lines",
        line=dict(color="#1f77b4", width=3), name="Efficient frontier"
    ))
    fig.add_trace(go.Scatter(
        x=asset_vols, y=asset_rets, mode="markers+text",
        marker=dict(size=14, color="black", symbol="x"),
        text=asset_labels, textposition="top center",
        name="Constituents", hovertemplate="%{text}<br>σ=%{x:.2%}<br>μ=%{y:.2%}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=[ms_sharpe_vol], y=[ms_sharpe_ret], mode="markers",
        marker=dict(size=18, color="gold", symbol="star",
                    line=dict(color="black", width=1.5)),
        name="Max Sharpe", hovertemplate="Max Sharpe<br>σ=%{x:.2%}<br>μ=%{y:.2%}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=[mv_vol], y=[mv_ret], mode="markers",
        marker=dict(size=18, color="red", symbol="diamond",
                    line=dict(color="black", width=1.5)),
        name="Min Volatility", hovertemplate="Min Vol<br>σ=%{x:.2%}<br>μ=%{y:.2%}<extra></extra>"
    ))
    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Annualised volatility (σ)",
        yaxis_title="Annualised expected return (μ)",
        template="plotly_white", height=560,
        xaxis=dict(tickformat=".0%"), yaxis=dict(tickformat=".0%"),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


# ---------------------------------------------------------------------------
# Weight allocations
# ---------------------------------------------------------------------------
def weights_bar(weights: np.ndarray, labels: list[str], title: str) -> go.Figure:
    order = np.argsort(weights)[::-1]
    fig = go.Figure(go.Bar(
        x=[labels[i] for i in order], y=weights[order] * 100,
        text=[f"{w:.1%}" for w in weights[order]],
        textposition="outside", marker_color="#1f77b4",
    ))
    fig.update_layout(title=title, yaxis_title="Weight (%)",
                      yaxis=dict(range=[0, max(weights) * 100 * 1.2 + 5]),
                      template="plotly_white", height=380)
    return fig


def weights_pie(weights: np.ndarray, labels: list[str], title: str) -> go.Figure:
    mask = weights > 1e-3
    fig = go.Figure(go.Pie(
        labels=[l for l, m in zip(labels, mask) if m],
        values=[w for w, m in zip(weights, mask) if m], hole=0.45,
        textinfo="label+percent"
    ))
    fig.update_layout(title=title, template="plotly_white", height=380)
    return fig


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
def cumulative_returns_chart(port_returns: pd.Series, bench_returns: pd.Series | None = None) -> go.Figure:
    port_eq = (1.0 + port_returns).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=port_eq.index, y=port_eq, name="Portfolio", line=dict(width=2.4)))
    if bench_returns is not None:
        bench_eq = (1.0 + bench_returns).cumprod()
        fig.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq, name="Benchmark", line=dict(dash="dash")))
    fig.update_layout(title="Cumulative growth of $1", yaxis_title="Equity ($)",
                      hovermode="x unified", template="plotly_white", height=420)
    return fig


def drawdown_chart(port_returns: pd.Series) -> go.Figure:
    eq = (1.0 + port_returns).cumprod()
    dd = eq / eq.cummax() - 1.0
    fig = go.Figure(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy",
                               line=dict(color="crimson"), name="Drawdown"))
    fig.update_layout(title="Drawdown (%)", yaxis_title="Drawdown (%)",
                      template="plotly_white", height=320)
    return fig


def monthly_heatmap(port_returns: pd.Series) -> go.Figure:
    monthly = (1.0 + port_returns).resample("M").prod() - 1.0
    pivot = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month_name().str[:3],
        "ret": monthly.values * 100,
    }).pivot(index="year", columns="month", values="ret")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=months)
    fig = px.imshow(pivot, text_auto=".1f", aspect="auto",
                    color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                    title="Monthly returns (%)")
    fig.update_layout(template="plotly_white", height=320)
    return fig


def return_distribution(port_returns: pd.Series, var95: float, cvar95: float) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram", "QQ-style empirical"))
    fig.add_trace(go.Histogram(x=port_returns * 100, nbinsx=60, marker_color="#1f77b4",
                               opacity=0.85, name="Returns"), row=1, col=1)
    fig.add_vline(x=-var95 * 100, line_dash="dash", line_color="orange", row=1, col=1)
    fig.add_vline(x=-cvar95 * 100, line_dash="dash", line_color="red", row=1, col=1)
    sorted_r = np.sort(port_returns) * 100
    fig.add_trace(go.Scatter(x=sorted_r, y=np.linspace(0, 1, len(sorted_r)),
                             line=dict(color="#1f77b4"), name="CDF"), row=1, col=2)
    fig.update_layout(template="plotly_white", height=380, showlegend=False)
    fig.update_xaxes(title_text="Daily return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=1, col=1)
    fig.update_xaxes(title_text="Return (%)", row=1, col=2)
    fig.update_yaxes(title_text="F(r)", row=1, col=2)
    return fig


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def monte_carlo_paths(paths: np.ndarray, n_show: int = 80, n_highlight: int = 5, seed: int = 7) -> go.Figure:
    """Path fan chart.

    Renders ``n_show`` paths at low opacity, ``n_highlight`` random paths at full opacity
    so the viewer can trace plausible futures, plus median + 5/95 percentile bands.
    """
    show = min(n_show, paths.shape[0])
    fig = go.Figure()
    rng = np.random.default_rng(seed)
    if paths.shape[0] > n_highlight:
        highlight_idx = set(rng.choice(paths.shape[0], size=n_highlight, replace=False).tolist())
    else:
        highlight_idx = set()
    for i in range(show):
        if i in highlight_idx:
            fig.add_trace(go.Scatter(
                y=paths[i], mode="lines",
                line=dict(color="rgba(31,119,180,0.85)", width=1.5),
                name="Sample path", showlegend=(i == min(highlight_idx)),
                hovertemplate="Day %{x}<br>Value: $%{y:,.0f}<extra></extra>",
            ))
        else:
            fig.add_trace(go.Scatter(
                y=paths[i], mode="lines",
                line=dict(color="rgba(31,119,180,0.10)", width=1),
                showlegend=False, hoverinfo="skip",
            ))
    p50 = np.median(paths, axis=0)
    p5, p95 = np.percentile(paths, [5, 95], axis=0)
    fig.add_trace(go.Scatter(y=p50, mode="lines", line=dict(color="#1f77b4", width=3), name="Median"))
    fig.add_trace(go.Scatter(y=p95, mode="lines", line=dict(color="green", dash="dash"), name="95th pct"))
    fig.add_trace(go.Scatter(y=p5, mode="lines", line=dict(color="red", dash="dash"), name="5th pct"))
    fig.update_layout(
        title=f"Monte Carlo equity paths ({show} of {paths.shape[0]} shown, {n_highlight} highlighted)",
        xaxis_title="Trading days", yaxis_title="Portfolio value ($)",
        template="plotly_white", height=460,
    )
    return fig


def monte_carlo_terminal_hist(terminal: np.ndarray, initial: float) -> go.Figure:
    pnl = (terminal - initial) * 100
    var5 = float(-np.quantile(terminal - initial, 0.05)) * 100
    cvar5 = float(-(terminal - initial)[(terminal - initial) <= -var5 / 100].mean()) * 100
    prob_loss = float((pnl < 0).mean()) * 100
    fig = go.Figure(go.Histogram(x=pnl, nbinsx=80, marker_color="#1f77b4", opacity=0.85))
    fig.add_vline(x=0, line_color="black", line_width=1.5,
                  annotation_text=f"P(loss) = {prob_loss:.1f}%",
                  annotation_position="top left",
                  annotation_font_size=13, annotation_font_color="black")
    fig.add_vline(x=-var5, line_color="orange", line_dash="dash",
                  annotation_text=f"VaR 95% = ${var5:.0f}", annotation_position="top right")
    fig.add_vline(x=-cvar5, line_color="red", line_dash="dash",
                  annotation_text=f"CVaR 95% = ${cvar5:.0f}", annotation_position="bottom right")
    fig.update_layout(title="Distribution of terminal P&L ($)",
                      xaxis_title="Profit / Loss ($)", yaxis_title="Frequency",
                      template="plotly_white", height=420)
    return fig


# ---------------------------------------------------------------------------
# Donut chart for live weight editing
# ---------------------------------------------------------------------------
def weight_donut(weights: np.ndarray, labels: list[str], title: str = "Current weights") -> go.Figure:
    """Donut chart that hides weights below 1% so small positions don't clutter."""
    pairs = sorted(zip(labels, weights), key=lambda p: -p[1])
    lbls = [l for l, _ in pairs if _ >= 0.01]
    vals = [w for _, w in pairs if w >= 0.01]
    fig = go.Figure(go.Pie(
        labels=lbls, values=vals, hole=0.55,
        textinfo="label+percent", textposition="outside",
        marker=dict(line=dict(color="white", width=2)),
    ))
    fig.update_layout(title=title, template="plotly_white", height=380,
                      showlegend=False, margin=dict(t=60, b=40, l=40, r=40))
    return fig


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------
def rolling_metric_chart(series_dict: dict[str, np.ndarray], index, title: str, ylabel: str) -> go.Figure:
    """Multi-line rolling chart with a horizontal zero/benchmark reference line."""
    fig = go.Figure()
    for name, arr in series_dict.items():
        fig.add_trace(go.Scatter(
            x=index, y=arr, mode="lines", name=name,
            hovertemplate=f"{name}<br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}<extra></extra>",
        ))
    fig.add_hline(y=0, line_color="black", line_width=0.7)
    fig.update_layout(
        title=title, yaxis_title=ylabel, xaxis_title="Date",
        template="plotly_white", height=320, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def backtest_compare_chart(series_dict: dict[str, pd.Series], title: str = "Historical backtest") -> go.Figure:
    """Cumulative equity chart comparing multiple strategies on the same axes."""
    fig = go.Figure()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, (name, s) in enumerate(series_dict.items()):
        eq = (1.0 + s).cumprod()
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq, mode="lines", name=name,
            line=dict(color=palette[i % len(palette)], width=2.2),
        ))
    fig.update_layout(
        title=title, yaxis_title="Equity ($1 \u2192 ?)", xaxis_title="Date",
        template="plotly_white", height=380, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig