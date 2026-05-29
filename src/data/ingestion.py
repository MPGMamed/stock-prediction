import yfinance as yf
import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fetch_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    save: bool = True,
    data_dir: str = "data",
) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    # yfinance >=0.2.x may return MultiIndex columns like ('Close', 'AAPL') — flatten to scalar
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    df.dropna(inplace=True)

    if save:
        Path(data_dir).mkdir(exist_ok=True)
        df.to_csv(f"{data_dir}/{ticker}.csv")

    return df


def fetch_multiple_tickers(
    tickers: list[str],
    start_date: str,
    end_date: str,
    data_dir: str = "data",
) -> dict[str, pd.DataFrame]:
    return {
        ticker: fetch_stock_data(ticker, start_date, end_date, save=True, data_dir=data_dir)
        for ticker in tickers
    }


def load_local_data(ticker: str, data_dir: str = "data") -> pd.DataFrame:
    path = f"{data_dir}/{ticker}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No local data for {ticker}. Run fetch_stock_data first.")
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


def get_market_data(ticker: str, config: dict) -> pd.DataFrame:
    """Try loading local data first; fall back to yfinance download."""
    try:
        return load_local_data(ticker, config["paths"]["data_dir"])
    except FileNotFoundError:
        return fetch_stock_data(
            ticker,
            config["data"]["start_date"],
            config["data"]["end_date"],
            data_dir=config["paths"]["data_dir"],
        )


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["daily_return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1
    return df


def train_test_split_timeseries(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    test_idx = int(n * (1 - test_size))
    val_idx = int(n * (1 - test_size - val_size))

    train = df.iloc[:val_idx]
    val = df.iloc[val_idx:test_idx]
    test = df.iloc[test_idx:]

    return train, val, test
