"""
simulation.py - Monte Carlo simulators for portfolio analysis.

* ``gbm_terminal``  - one-shot Geometric Brownian Motion, used for VaR/CVaR.
* ``gbm_paths``     - full equity paths for visualisation.
* ``cholesky_shocks`` - correlated normal shocks shared by both simulators.
"""
from __future__ import annotations

import numpy as np


def cholesky_shocks(n_sims: int, n_steps: int, cov: np.ndarray, seed: int | None = None) -> np.ndarray:
    """(n_sims, n_steps, n_assets) array of correlated standard normals."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(cov.shape[0]))
    Z = rng.standard_normal(size=(n_sims, n_steps, cov.shape[0]))
    return Z @ L.T


def gbm_terminal(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    horizon_days: int = 252,
    n_sims: int = 10_000,
    initial_value: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Distribution of terminal portfolio values via correlated GBM."""
    shocks = cholesky_shocks(n_sims, horizon_days, cov, seed=seed)
    drift = (mu - 0.5 * np.diag(cov)) / 252.0
    log_inc = drift + shocks / np.sqrt(252.0)
    log_path = np.cumsum(log_inc @ np.asarray(weights, dtype=float), axis=1)
    return initial_value * np.exp(log_path[:, -1])


def gbm_paths(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    horizon_days: int = 252,
    n_sims: int = 200,
    initial_value: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """(n_sims, horizon_days+1) array of portfolio paths starting at initial_value."""
    shocks = cholesky_shocks(n_sims, horizon_days, cov, seed=seed)
    drift = (mu - 0.5 * np.diag(cov)) / 252.0
    log_inc = drift + shocks / np.sqrt(252.0)
    log_path = np.cumsum(log_inc @ np.asarray(weights, dtype=float), axis=1)
    paths = initial_value * np.exp(np.concatenate([np.zeros((n_sims, 1)), log_path], axis=1))
    return paths


def simulate_var(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    horizon_days: int = 252,
    n_sims: int = 10_000,
    alpha: float = 0.05,
    initial_value: float = 1.0,
    seed: int | None = None,
) -> dict:
    """VaR / CVaR on the simulated terminal value distribution."""
    terminal = gbm_terminal(mu, cov, weights, horizon_days, n_sims, initial_value, seed)
    pnl = terminal - initial_value
    var = float(-np.quantile(pnl, alpha))
    tail = pnl[pnl <= -var]
    cvar = float(-tail.mean()) if len(tail) else var
    return {
        "VaR": var,
        "CVaR": cvar,
        "mean": float(pnl.mean()),
        "std": float(pnl.std()),
        "terminal_values": terminal,
        "pnl": pnl,
    }