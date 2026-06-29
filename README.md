# AssetAllocator

> Interactive Modern Portfolio Theory dashboard with live market data, efficient frontier, risk analytics, and Monte Carlo simulation.

A quant-grade Streamlit app that pulls real prices from Yahoo Finance, optimises portfolio weights with SciPy, computes the full risk metric stack (VaR, CVaR, Sharpe, Sortino, Calmar, drawdown), and simulates correlated Geometric Brownian Motion paths for forward-looking risk assessment.

## Why this project

Production-style end-to-end workflow:
- Real data pipeline (yfinance + SQLite cache)
- Modular `src/` layout (`data`, `calculations`, `simulation`, `visualizations`)
- Multi-page Streamlit app with sidebar-driven configuration
- 18 pytest assertions covering optimisation, risk, and Monte Carlo
- Docker-ready

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

Then open <http://localhost:8501>.

### Docker

```bash
docker compose up --build
```

### Tests

```bash
pytest -q
```

## Features

| Tab | What it does |
|---|---|
| **Home** | Normalised price history + global sidebar config (tickers, period, RF rate, weight constraints) |
| **Efficient Frontier** | Markowitz MVP + Max-Sharpe portfolios, frontier curve, random-portfolio cloud, weight tables |
| **Risk Dashboard** | Historical VaR/CVaR (95/99%), drawdown, return distribution with tail markers, correlation heatmap, user-editable weights |
| **Performance** | Cumulative growth vs benchmark, monthly return heatmap, full risk metric table |
| **Monte Carlo** | Correlated GBM path simulation (Cholesky factorisation), P&L distribution, configurable VaR |

## Repo structure

```
AssetAllocator/
  app/
    main.py                    # sidebar + data load
    pages/
      1_Efficient_Frontier.py
      2_Risk_Dashboard.py
      3_Performance.py
      4_Monte_Carlo.py
  src/
    data.py                    # yfinance loader + SQLite cache + synthetic fallback
    calculations.py            # MPT optimisers + risk metrics (numpy/scipy only)
    simulation.py              # Cholesky GBM simulator for Monte Carlo
    visualizations.py          # Plotly figure factories
  tests/
    test_calculations.py
    test_data.py
    test_simulation.py
  Dockerfile
  docker-compose.yml
  requirements.txt
  README.md
```

## Math notes

### Mean-Variance optimisation

Markowitz (1952). For an $n$-asset portfolio with weight vector $w$:

$$
\max_w \; \mu^\top w - \tfrac{\lambda}{2} w^\top \Sigma w
\quad \text{s.t.} \quad \mathbf{1}^\top w = 1,\; w_i \in [0, w_{\max}]
$$

Solved with SciPy's `SLSQP`. The efficient frontier is traced by sweeping target returns and re-solving at each step.

### Risk metrics

- **Sharpe**: $(\mu_p - r_f)/\sigma_p$ (annualised)
- **Sortino**: same numerator, downside deviation only
- **Calmar**: CAGR / |max drawdown|
- **VaR**: $\alpha$-quantile of daily return distribution (positive loss number)
- **CVaR** (Expected Shortfall): conditional tail mean beyond VaR

### Monte Carlo

Correlated GBM with Cholesky factorisation of the annualised covariance matrix:

$$
\Delta \log V_t = \left(\mu - \tfrac{1}{2}\operatorname{diag}(\Sigma)\right)\Delta t + L\,Z_t,\; L L^\top = \Sigma
$$

Output: terminal portfolio values after `horizon` days, plus per-sim P&L.

## Data flow

```
yfinance → SQLite cache (data/cache.db) → prices (wide DataFrame)
   ↓
returns = pct_change() → mu, cov (annualised)
   ↓
optimise + simulate → Plotly figures → Streamlit
```

If `yfinance` fails (offline / blocked), the app falls back to a correlated synthetic panel so it still runs.

## Configuration

Edit `app/main.py` `DEFAULT_TICKERS` for a different default universe, or pass tickers via the sidebar text box.

## License

MIT