import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = sns.color_palette("husl", 8)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


# ── Matplotlib / Seaborn ────────────────────────────────────────────────────

def plot_price_history(df: pd.DataFrame, ticker: str = "Stock") -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    ax1, ax2, ax3 = axes

    ax1.plot(df.index, df["close"], label="Close", color=PALETTE[0], linewidth=1.2)
    for col, color in zip(["sma_20", "sma_50", "sma_200"], PALETTE[1:4]):
        if col in df.columns:
            ax1.plot(df.index, df[col], label=col.upper(), color=color, linewidth=0.8, alpha=0.8)
    ax1.set_ylabel("Price ($)")
    ax1.set_title(f"{ticker} Price History with Moving Averages")
    ax1.legend(loc="upper left", fontsize=8)

    ax2.bar(df.index, df["volume"], color=PALETTE[4], alpha=0.6, width=1)
    if "vol_sma_20" in df.columns:
        ax2.plot(df.index, df["vol_sma_20"], color=PALETTE[5], linewidth=0.9)
    ax2.set_ylabel("Volume")

    if "daily_return" in df.columns:
        returns = df["daily_return"].fillna(0)
        colors = [PALETTE[0] if r >= 0 else PALETTE[6] for r in returns]
        ax3.bar(df.index, returns * 100, color=colors, width=1, alpha=0.7)
        ax3.axhline(0, color="black", linewidth=0.5)
        ax3.set_ylabel("Daily Return (%)")

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    return fig


def plot_returns_distribution(df: pd.DataFrame, ticker: str = "Stock") -> plt.Figure:
    returns = df["daily_return"].dropna()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sns.histplot(returns * 100, bins=80, kde=True, ax=axes[0], color=PALETTE[0])
    axes[0].axvline(returns.mean() * 100, color="red", linestyle="--", label=f"Mean: {returns.mean()*100:.2f}%")
    axes[0].set_title("Daily Returns Distribution")
    axes[0].set_xlabel("Return (%)")
    axes[0].legend()

    from scipy import stats
    stats.probplot(returns, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot (vs. Normal)")

    rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
    axes[2].plot(rolling_vol.index, rolling_vol, color=PALETTE[2], linewidth=0.9)
    axes[2].set_title("30-Day Rolling Annualized Volatility")
    axes[2].set_ylabel("Volatility (%)")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle(f"{ticker} Return Analysis", fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(dfs: dict[str, pd.DataFrame]) -> plt.Figure:
    closes = pd.DataFrame({ticker: df["close"] for ticker, df in dfs.items()})
    returns = closes.pct_change().dropna()
    corr = returns.corr()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                ax=axes[0], square=True, linewidths=0.5)
    axes[0].set_title("Return Correlation Matrix")

    returns_norm = (returns - returns.mean()) / returns.std()
    cum_returns = (1 + returns).cumprod()
    for i, col in enumerate(cum_returns.columns):
        axes[1].plot(cum_returns.index, cum_returns[col], label=col, color=PALETTE[i])
    axes[1].set_title("Cumulative Returns Comparison")
    axes[1].set_ylabel("Growth of $1")
    axes[1].legend()
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    return fig


def plot_technical_indicators(df: pd.DataFrame, ticker: str = "Stock") -> plt.Figure:
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(df.index, df["close"], label="Close", color=PALETTE[0])
    if "bb_upper" in df.columns:
        axes[0].fill_between(df.index, df["bb_lower"], df["bb_upper"], alpha=0.15, color=PALETTE[1])
        axes[0].plot(df.index, df["bb_middle"], "--", color=PALETTE[1], linewidth=0.8, label="BB Middle")
    axes[0].set_title(f"{ticker} Technical Indicators")
    axes[0].set_ylabel("Price ($)")
    axes[0].legend(fontsize=8)

    if "rsi" in df.columns:
        axes[1].plot(df.index, df["rsi"], color=PALETTE[2])
        axes[1].axhline(70, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        axes[1].axhline(30, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
        axes[1].set_ylabel("RSI")
        axes[1].set_ylim(0, 100)

    if "macd" in df.columns:
        axes[2].plot(df.index, df["macd"], label="MACD", color=PALETTE[3])
        axes[2].plot(df.index, df["macd_signal"], label="Signal", color=PALETTE[4])
        colors = [PALETTE[0] if v >= 0 else PALETTE[6] for v in df["macd_hist"]]
        axes[2].bar(df.index, df["macd_hist"], color=colors, alpha=0.6, width=1)
        axes[2].set_ylabel("MACD")
        axes[2].legend(fontsize=8)

    if "atr" in df.columns:
        axes[3].plot(df.index, df["atr"], color=PALETTE[5])
        axes[3].set_ylabel("ATR")

    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20, title: str = "") -> plt.Figure:
    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top["feature"][::-1], top["importance"][::-1], color=PALETTE[0])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances — {title}")
    for bar, val in zip(bars, top["importance"][::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: list = None) -> plt.Figure:
    labels = labels or ["Down", "Up"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_equity_curves(backtest_df: pd.DataFrame, ticker: str = "Stock") -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(backtest_df.index, backtest_df["equity_strategy"] * 100 - 100,
                 label="Strategy", color=PALETTE[0], linewidth=1.2)
    axes[0].plot(backtest_df.index, backtest_df["equity_bah"] * 100 - 100,
                 label="Buy & Hold", color=PALETTE[3], linewidth=1.2, linestyle="--")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel("Return (%)")
    axes[0].set_title(f"{ticker} Backtest: Strategy vs. Buy & Hold")
    axes[0].legend()
    axes[0].fill_between(backtest_df.index,
                          backtest_df["equity_strategy"] * 100 - 100,
                          0, alpha=0.1, color=PALETTE[0])

    dd_strat = backtest_df["equity_strategy"] / backtest_df["equity_strategy"].cummax() - 1
    dd_bah = backtest_df["equity_bah"] / backtest_df["equity_bah"].cummax() - 1
    axes[1].fill_between(backtest_df.index, dd_strat * 100, 0, alpha=0.4, color=PALETTE[0], label="Strategy DD")
    axes[1].fill_between(backtest_df.index, dd_bah * 100, 0, alpha=0.2, color=PALETTE[3], label="B&H DD")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].legend()
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    return fig


def plot_prediction_vs_actual(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "Predictions vs Actuals",
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    axes[0].plot(dates, actual, label="Actual", color=PALETTE[0], linewidth=1.2)
    axes[0].plot(dates, predicted, label="Predicted", color=PALETTE[5], linewidth=1.0, linestyle="--")
    axes[0].set_title(title)
    axes[0].set_ylabel("Price ($)")
    axes[0].legend()

    residuals = actual - predicted
    axes[1].bar(dates, residuals, color=[PALETTE[0] if r >= 0 else PALETTE[6] for r in residuals], alpha=0.6, width=1)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Residual")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    return fig


# ── Plotly (interactive) ─────────────────────────────────────────────────────

def plotly_candlestick(df: pd.DataFrame, ticker: str = "Stock") -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="OHLC",
    ), row=1, col=1)

    for col, color in [("sma_20", "blue"), ("sma_50", "orange")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col.upper(),
                                      line=dict(color=color, width=1)), row=1, col=1)

    colors = ["green" if r >= 0 else "red" for r in df["close"].diff().fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], marker_color=colors,
                          name="Volume", opacity=0.5), row=2, col=1)

    fig.update_layout(
        title=f"{ticker} Interactive Chart",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=600,
    )
    return fig
