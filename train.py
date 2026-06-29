from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import numpy as np
from src.data import make_synthetic, prices_to_returns, estimate_mu_cov
from src.calculations import max_sharpe_portfolio, min_volatility_portfolio, portfolio_summary
from src.persist import save_model
from src.evaluate import save_metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-days", type=int, default=756)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    print("Generating synthetic price data ...")
    prices = make_synthetic(n_days=a.n_days, seed=a.seed)
    returns = prices_to_returns(prices)
    mu, cov = estimate_mu_cov(returns)

    n_assets = len(prices.columns)
    print(f"  Assets: {n_assets}")
    print(f"  Trading days: {len(prices)}")
    print(f"  Annualised mu: {dict(zip(prices.columns, np.round(mu, 4)))}")

    w_max_sharpe = max_sharpe_portfolio(mu, cov, rf=0.04)
    w_min_vol = min_volatility_portfolio(mu, cov)

    model = {
        "prices": prices,
        "returns": returns,
        "mu": mu,
        "cov": cov,
        "tickers": list(prices.columns),
        "max_sharpe_weights": w_max_sharpe,
        "min_vol_weights": w_min_vol,
    }
    save_model(model)
    print("  Model saved to models/model.pkl")

    ms_summary = portfolio_summary(returns, w_max_sharpe, rf=0.04)
    mv_summary = portfolio_summary(returns, w_min_vol, rf=0.04)
    metrics = {
        "n_assets": n_assets,
        "n_days": len(prices),
        "max_sharpe": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in ms_summary.items()},
        "min_volatility": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in mv_summary.items()},
    }
    save_metrics(metrics)
    print("  Metrics saved to models/metrics.json")
    print("Done.")


if __name__ == "__main__":
    main()
