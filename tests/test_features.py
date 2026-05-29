import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd
import numpy as np
from src.features.engineering import (
    add_moving_averages, add_rsi, add_macd, add_bollinger_bands,
    add_atr, add_target, get_feature_columns, adf_test,
)


def make_df(n: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    np.random.seed(42)
    close = 200 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close * 0.99,
        "high": close * 1.015,
        "low": close * 0.985,
        "close": close,
        "volume": np.random.randint(5_000_000, 20_000_000, n),
    }, index=idx)


def test_moving_averages_shape():
    df = make_df()
    result = add_moving_averages(df, windows=[5, 20])
    for col in ["sma_5", "sma_20", "ema_5", "ema_20"]:
        assert col in result.columns


def test_rsi_range():
    df = make_df()
    result = add_rsi(df, period=14).dropna()
    assert result["rsi"].between(0, 100).all()


def test_macd_columns():
    df = make_df()
    result = add_macd(df)
    assert {"macd", "macd_signal", "macd_hist"}.issubset(result.columns)


def test_bollinger_bands():
    df = make_df()
    result = add_bollinger_bands(df).dropna()
    assert (result["bb_upper"] >= result["bb_middle"]).all()
    assert (result["bb_middle"] >= result["bb_lower"]).all()


def test_atr_positive():
    df = make_df()
    result = add_atr(df).dropna()
    assert (result["atr"] > 0).all()


def test_target_binary():
    df = make_df()
    result = add_target(df, horizon=1).dropna()
    assert set(result["target_direction"].unique()).issubset({0, 1})


def test_feature_columns_excludes_ohlcv():
    from src.data.ingestion import compute_returns
    from src.features.engineering import build_feature_matrix
    import yaml
    with open("config/config.yaml") as f:
        import yaml
        config = yaml.safe_load(f)
    df = make_df()
    df = compute_returns(df)
    feat_df = build_feature_matrix(df, config)
    fcols = get_feature_columns(feat_df)
    for excluded in ["open", "high", "low", "close", "volume"]:
        assert excluded not in fcols


def test_adf_test_stationary():
    series = pd.Series(np.random.randn(200))
    result = adf_test(series)
    assert result["is_stationary"] == True
    assert "p_value" in result
