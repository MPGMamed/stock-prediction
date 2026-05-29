# Stock Market Analysis & Prediction Pipeline

An end-to-end machine learning pipeline for stock market analysis and direction prediction, combining classical ML models with deep learning and a REST API for serving predictions.

---

## Project Structure

```
stock-prediction/
├── config/
│   └── config.yaml              # All hyperparameters & paths
├── data/                        # Raw CSV data (gitignored)
├── models_saved/                # Serialised model artefacts (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   └── 02_modeling.ipynb        # Model training, evaluation & backtesting
├── src/
│   ├── data/
│   │   └── ingestion.py         # yfinance download, returns, train/val/test split
│   ├── features/
│   │   └── engineering.py       # 40+ technical indicators + lag/volatility features
│   ├── models/
│   │   ├── random_forest.py     # Scikit-learn RF wrapper with GridSearchCV
│   │   ├── xgboost_model.py     # XGBoost wrapper with early stopping
│   │   └── lstm.py              # TensorFlow LSTM with sequence builder & scaler
│   ├── evaluation/
│   │   └── metrics.py           # Classification/regression metrics, Sharpe, backtester
│   └── visualization/
│       └── plots.py             # Matplotlib/Seaborn + Plotly interactive charts
├── api/
│   ├── main.py                  # FastAPI app (predict / metrics / health endpoints)
│   └── schemas.py               # Pydantic request/response models
├── tests/
│   ├── test_ingestion.py
│   ├── test_features.py
│   └── test_models.py
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the EDA notebook

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 3. Train models and backtest

```bash
jupyter notebook notebooks/02_modeling.ipynb
```

### 4. Start the prediction API

```bash
uvicorn api.main:app --reload
```

API docs available at **http://127.0.0.1:8000/docs**

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Features

### Data Pipeline
- Automatic download via **yfinance** for any ticker
- Configurable date ranges and multiple tickers
- Time-series aware train / validation / test split (no data leakage)

### Feature Engineering (40+ features)
| Category | Features |
|---|---|
| Trend | SMA & EMA (5, 10, 20, 50, 200) |
| Momentum | RSI(14), MACD, MACD Signal, Histogram |
| Volatility | Bollinger Bands (width, %B), ATR, Rolling Vol |
| Volume | OBV, Volume SMA ratio |
| Price patterns | Body size, shadows, is-bullish |
| Lag features | Close & return lags 1, 2, 3, 5, 10 days |
| Statistical | Stationarity (ADF test), rolling skew/kurtosis |

### Models
| Model | Task | Key library |
|---|---|---|
| Random Forest | Direction classification | scikit-learn |
| XGBoost | Direction classification | xgboost |
| LSTM | Price regression | TensorFlow/Keras |

All tree models support:
- **GridSearchCV** with `TimeSeriesSplit` for leak-free tuning
- Feature importance ranking
- Save/load via `joblib`

LSTM supports:
- Multi-layer architecture with BatchNorm + Dropout
- Early stopping & learning-rate scheduler
- Inverse-transform predictions back to price scale

### Evaluation
- Classification: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- Regression: RMSE, MAE, R², MAPE, Directional Accuracy
- Backtesting: Sharpe Ratio, Max Drawdown, Total Return, Buy-and-Hold comparison

### REST API (FastAPI)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health + loaded models |
| `/predict` | POST | Direction prediction with confidence |
| `/metrics/{ticker}/{model}` | GET | Live test-set metrics |
| `/tickers` | GET | Available tickers |

**Example request:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "model": "xgboost", "horizon_days": 1}'
```

**Example response:**
```json
{
  "ticker": "AAPL",
  "model": "xgboost",
  "direction": "UP",
  "confidence": 0.67,
  "horizon_days": 1,
  "last_close": 189.43,
  "features_used": 54
}
```

---

## Configuration

All parameters live in `config/config.yaml`:

```yaml
data:
  tickers: ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
  start_date: "2018-01-01"

models:
  xgboost:
    n_estimators: 200
    learning_rate: 0.05
    ...
```

---

## Tech Stack

`Python 3.11` · `pandas` · `numpy` · `scikit-learn` · `xgboost` · `TensorFlow` · `FastAPI` · `Pydantic` · `yfinance` · `statsmodels` · `matplotlib` · `seaborn` · `plotly`
