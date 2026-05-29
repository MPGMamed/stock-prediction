import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        p = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
        metrics["roc_auc"] = roc_auc_score(y_true, p)
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100,
    }


def directional_accuracy(y_true_returns: np.ndarray, y_pred_returns: np.ndarray) -> float:
    return np.mean(np.sign(y_true_returns) == np.sign(y_pred_returns))


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0, annualize: int = 252) -> float:
    excess = returns - risk_free / annualize
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(annualize)


def max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()


def backtest_strategy(
    prices: pd.Series,
    signals: np.ndarray,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """
    Simple long/flat backtest. signals: 1 = long, 0 = flat.
    Returns a DataFrame with strategy and buy-and-hold equity curves.
    """
    df = pd.DataFrame(index=prices.index)
    df["price"] = prices.values
    df["signal"] = signals
    df["daily_return"] = prices.pct_change()
    df["strategy_return"] = df["signal"].shift(1) * df["daily_return"]

    trade_changes = df["signal"].diff().abs()
    df["strategy_return"] -= trade_changes * transaction_cost

    df["equity_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["equity_bah"] = (1 + df["daily_return"]).cumprod()
    df.dropna(inplace=True)
    return df


def strategy_summary(backtest_df: pd.DataFrame) -> dict:
    strat_returns = backtest_df["strategy_return"].values
    bah_returns = backtest_df["daily_return"].values
    return {
        "strategy_sharpe": sharpe_ratio(strat_returns),
        "bah_sharpe": sharpe_ratio(bah_returns),
        "strategy_max_dd": max_drawdown(backtest_df["equity_strategy"].values),
        "bah_max_dd": max_drawdown(backtest_df["equity_bah"].values),
        "strategy_total_return": backtest_df["equity_strategy"].iloc[-1] - 1,
        "bah_total_return": backtest_df["equity_bah"].iloc[-1] - 1,
        "n_trades": int(backtest_df["signal"].diff().abs().sum()),
    }


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(results).T.round(4)
