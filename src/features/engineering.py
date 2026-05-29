import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller


def add_moving_averages(df: pd.DataFrame, windows: list[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    df["bb_upper"] = sma + num_std * std
    df["bb_middle"] = sma
    df["bb_lower"] = sma - num_std * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(period).mean()
    return df


def add_volume_features(df: pd.DataFrame, windows: list[int] = [5, 20]) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"vol_sma_{w}"] = df["volume"].rolling(w).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma_20"].replace(0, np.nan)
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
    return df


def add_price_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_range"] = (df["high"] - df["low"]) / df["close"]
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]
    df["body_size"] = (df["close"] - df["open"]).abs() / df["close"]
    df["is_bullish"] = (df["close"] > df["open"]).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, lags: list[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"return_lag_{lag}"] = df["close"].pct_change().shift(lag)
    return df


def add_volatility_features(df: pd.DataFrame, windows: list[int] = [5, 10, 20]) -> pd.DataFrame:
    df = df.copy()
    returns = df["close"].pct_change()
    for w in windows:
        df[f"volatility_{w}"] = returns.rolling(w).std() * np.sqrt(252)
    return df


def add_target(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> pd.DataFrame:
    """
    Binary classification target: 1 if price goes up by more than threshold in `horizon` days.
    Also adds a regression target: future return.
    """
    df = df.copy()
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    df["target_return"] = future_return
    df["target_direction"] = (future_return > threshold).astype(int)
    return df


def build_feature_matrix(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    cfg_f = config["features"]
    df = add_moving_averages(df, windows=cfg_f["window_sizes"])
    df = add_rsi(df, period=cfg_f["rsi_period"])
    df = add_macd(df, fast=cfg_f["macd_fast"], slow=cfg_f["macd_slow"], signal=cfg_f["macd_signal"])
    df = add_bollinger_bands(df, period=cfg_f["bb_period"], num_std=cfg_f["bb_std"])
    df = add_atr(df)
    df = add_volume_features(df)
    df = add_price_patterns(df)
    df = add_lag_features(df)
    df = add_volatility_features(df)
    df = add_target(df)
    df.dropna(inplace=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"open", "high", "low", "close", "volume", "target_return", "target_direction"}
    return [c for c in df.columns if c not in exclude]


def adf_test(series: pd.Series) -> dict:
    result = adfuller(series.dropna())
    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }
