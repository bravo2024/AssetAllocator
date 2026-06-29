"""Tests for data layer: synthetic panel + return conversion."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import (
    ASSETS,
    estimate_mu_cov,
    make_synthetic,
    prices_to_returns,
)


def test_synthetic_panel_shape_and_columns():
    df = make_synthetic(n_days=300)
    assert df.shape == (300, len(ASSETS))
    assert list(df.columns) == ASSETS
    assert isinstance(df.index, pd.DatetimeIndex)
    assert (df > 0).all().all()  # prices are positive


def test_synthetic_deterministic():
    df1 = make_synthetic(n_days=200, seed=11)
    df2 = make_synthetic(n_days=200, seed=11)
    pd.testing.assert_frame_equal(df1, df2)


def test_synthetic_different_seeds_differ():
    df1 = make_synthetic(n_days=200, seed=1)
    df2 = make_synthetic(n_days=200, seed=2)
    assert not df1.equals(df2)


def test_prices_to_returns_drops_first_row():
    prices = make_synthetic(50)
    r = prices_to_returns(prices, method="arithmetic")
    assert len(r) == len(prices) - 1


def test_prices_to_returns_log_equivalence():
    prices = make_synthetic(50)
    r_simple = prices_to_returns(prices, method="arithmetic")
    r_log = prices_to_returns(prices, method="log")
    np.testing.assert_allclose(r_log, np.log1p(r_simple), atol=1e-9)


def test_estimate_mu_cov_shapes_and_finite():
    prices = make_synthetic(300)
    r = prices_to_returns(prices)
    mu, cov = estimate_mu_cov(r)
    assert mu.shape == (len(ASSETS),)
    assert cov.shape == (len(ASSETS), len(ASSETS))
    assert np.isfinite(mu).all()
    assert np.isfinite(cov).all()
    # Covariance is symmetric positive semi-definite
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(cov)
    assert (eigvals >= -1e-9).all()