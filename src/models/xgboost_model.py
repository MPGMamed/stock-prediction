import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from pathlib import Path


class StockXGBoost:
    def __init__(self, task: str = "classification", **kwargs):
        self.task = task
        params = {
            "n_estimators": kwargs.get("n_estimators", 200),
            "max_depth": kwargs.get("max_depth", 6),
            "learning_rate": kwargs.get("learning_rate", 0.05),
            "subsample": kwargs.get("subsample", 0.8),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
            "random_state": kwargs.get("random_state", 42),
            "n_jobs": -1,
            "verbosity": 0,
        }
        if task == "classification":
            self.model = XGBClassifier(**params, eval_metric="logloss", use_label_encoder=False)
        else:
            self.model = XGBRegressor(**params, eval_metric="rmse")

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None) -> "StockXGBoost":
        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["verbose"] = False
        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only for classification.")
        return self.model.predict_proba(X)

    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {"feature": feature_names, "importance": self.model.feature_importances_}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def tune_hyperparams(self, X: np.ndarray, y: np.ndarray, cv_splits: int = 5) -> dict:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.9],
        }
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        scoring = "f1" if self.task == "classification" else "neg_mean_squared_error"
        gs = GridSearchCV(self.model, param_grid, cv=tscv, scoring=scoring, n_jobs=-1, verbose=1)
        gs.fit(X, y)
        self.model = gs.best_estimator_
        return gs.best_params_

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str, task: str = "classification") -> "StockXGBoost":
        obj = cls.__new__(cls)
        obj.task = task
        obj.model = joblib.load(path)
        return obj
