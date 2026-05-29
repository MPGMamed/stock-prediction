from pydantic import BaseModel, Field
from typing import Literal


class PredictionRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"], description="Stock ticker symbol")
    model: Literal["random_forest", "xgboost"] = Field(
        "xgboost", description="Model to use for prediction"
    )
    horizon_days: int = Field(1, ge=1, le=30, description="Prediction horizon in days")


class PredictionResponse(BaseModel):
    ticker: str
    model: str
    direction: Literal["UP", "DOWN"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    horizon_days: int
    last_close: float
    features_used: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    version: str = "1.0.0"


class ModelMetricsResponse(BaseModel):
    model: str
    ticker: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None = None
