"""Tests for Monte Carlo simulation: GBM path sanity + VaR ordering."""
from __future__ import annotations

import numpy as np

from src.simulation import (
    cholesky_shocks,
    gbm_paths,
    gbm_terminal,
    simulate_var,
)


def test_cholesky_shocks_zero_correlation_uncorrelated():
    cov = np.eye(3)
    Z = cholesky_shocks(n_sims=5000, n_steps=5, cov=cov, seed=0)
    # Covariance across sims of each step-pair should be ~I
    flat = Z.reshape(-1, 3)
    emp_cov = np.cov(flat.T)
    np.testing.assert_allclose(emp_cov, np.eye(3), atol=0.05)


def test_gbm_paths_start_at_initial_value():
    cov = np.eye(2) * 0.04
    mu = np.array([0.08, 0.05])
    w = np.array([0.5, 0.5])
    paths = gbm_paths(mu, cov, w, horizon_days=50, n_sims=20, initial_value=1000.0, seed=1)
    np.testing.assert_allclose(paths[:, 0], 1000.0)


def test_gbm_terminal_positive_and_mean_above_one_for_positive_drift():
    cov = np.eye(2) * 0.02
    mu = np.array([0.10, 0.10])
    w = np.array([0.5, 0.5])
    terminal = gbm_terminal(mu, cov, w, horizon_days=252, n_sims=5000, initial_value=1.0, seed=0)
    assert (terminal > 0).all()
    assert terminal.mean() > 1.0  # positive drift => growth


def test_simulate_var_ordering_and_finite():
    cov = np.eye(2) * 0.04
    mu = np.array([0.08, 0.04])
    w = np.array([0.6, 0.4])
    out = simulate_var(mu, cov, w, horizon_days=252, n_sims=3000, alpha=0.05, initial_value=1.0, seed=2)
    assert out["CVaR"] >= out["VaR"]
    assert np.isfinite(out["mean"]) and np.isfinite(out["std"])