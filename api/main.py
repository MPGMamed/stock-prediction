"""
FastAPI service exposing stock direction prediction.

Run:  uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""

import os
import sys
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.schemas import PredictionRequest, PredictionResponse, HealthResponse, ModelMetricsResponse
from src.data.ingestion import load_config, get_market_data, compute_returns
from src.features.engineering import build_feature_matrix, get_feature_columns
from src.models.random_forest import StockRandomForest
from src.models.xgboost_model import StockXGBoost

CONFIG = load_config("config/config.yaml")
MODELS: dict = {}


def _load_or_train(model_name: str, ticker: str):
    key = f"{model_name}_{ticker}"
    if key in MODELS:
        return MODELS[key]

    rf_path = f"models_saved/{ticker}_rf.pkl"
    xgb_path = f"models_saved/{ticker}_xgb.pkl"

    df = get_market_data(ticker, CONFIG)
    df = compute_returns(df)
    df = build_feature_matrix(df, CONFIG)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["target_direction"].values

    if model_name == "random_forest":
        if os.path.exists(rf_path):
            model = StockRandomForest.load(rf_path)
        else:
            model = StockRandomForest(task="classification", **CONFIG["models"]["random_forest"])
            model.fit(X, y)
            model.save(rf_path)
    elif model_name == "xgboost":
        if os.path.exists(xgb_path):
            model = StockXGBoost.load(xgb_path)
        else:
            model = StockXGBoost(task="classification", **CONFIG["models"]["xgboost"])
            model.fit(X, y)
            model.save(xgb_path)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    MODELS[key] = (model, df, feature_cols)
    return MODELS[key]


@asynccontextmanager
async def lifespan(app: FastAPI):
    ticker = CONFIG["data"]["default_ticker"]
    for model_name in ["random_forest", "xgboost"]:
        _load_or_train(model_name, ticker)
    yield
    MODELS.clear()


app = FastAPI(
    title="Stock Prediction API",
    description="ML-powered stock direction prediction using Random Forest and XGBoost.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse(
        status="ok",
        models_loaded=list(MODELS.keys()),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    try:
        model, df, feature_cols = _load_or_train(request.model, request.ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latest = df[feature_cols].iloc[[-1]].values
    proba = model.predict_proba(latest)[0]
    direction_idx = int(np.argmax(proba))
    confidence = float(proba[direction_idx])

    return PredictionResponse(
        ticker=request.ticker.upper(),
        model=request.model,
        direction="UP" if direction_idx == 1 else "DOWN",
        confidence=confidence,
        horizon_days=request.horizon_days,
        last_close=float(df["close"].iloc[-1]),
        features_used=len(feature_cols),
    )


@app.get("/metrics/{ticker}/{model_name}", response_model=ModelMetricsResponse, tags=["Evaluation"])
def get_model_metrics(ticker: str, model_name: str):
    from sklearn.model_selection import train_test_split
    from src.evaluation.metrics import classification_metrics

    try:
        model, df, feature_cols = _load_or_train(model_name, ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    X = df[feature_cols].values
    y = df["target_direction"].values
    split = int(len(X) * 0.8)
    X_test, y_test = X[split:], y[split:]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    metrics = classification_metrics(y_test, y_pred, y_prob)

    return ModelMetricsResponse(
        model=model_name,
        ticker=ticker.upper(),
        **metrics,
    )


@app.get("/tickers", tags=["Data"])
def available_tickers():
    return {"tickers": CONFIG["data"]["tickers"]}
