import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.data.ingestion import compute_returns, train_test_split_timeseries


def make_df(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    np.random.seed(0)
    close = 150 + np.cumsum(np.random.randn(n))
    return pd.DataFrame({
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=idx)


def test_compute_returns_columns():
    df = make_df()
    result = compute_returns(df)
    assert "daily_return" in result.columns
    assert "log_return" in result.columns
    assert "cumulative_return" in result.columns


def test_compute_returns_values():
    df = make_df()
    result = compute_returns(df)
    expected = df["close"].pct_change()
    pd.testing.assert_series_equal(result["daily_return"], expected, check_names=False)


def test_train_test_split_sizes():
    df = make_df(200)
    train, val, test = train_test_split_timeseries(df, test_size=0.2, val_size=0.1)
    assert len(train) + len(val) + len(test) == len(df)
    assert len(test) == pytest.approx(40, abs=1)
    assert len(val) == pytest.approx(20, abs=1)


def test_train_test_split_no_leakage():
    df = make_df(200)
    train, val, test = train_test_split_timeseries(df)
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()
