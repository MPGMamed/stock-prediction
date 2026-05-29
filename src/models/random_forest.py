import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from pathlib import Path


class StockRandomForest:
    def __init__(self, task: str = "classification", **kwargs):
        self.task = task
        params = {
            "n_estimators": kwargs.get("n_estimators", 200),
            "max_depth": kwargs.get("max_depth", 10),
            "min_samples_split": kwargs.get("min_samples_split", 5),
            "random_state": kwargs.get("random_state", 42),
            "n_jobs": -1,
        }
        if task == "classification":
            self.model = RandomForestClassifier(**params, class_weight="balanced")
        else:
            self.model = RandomForestRegressor(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StockRandomForest":
        self.model.fit(X, y)
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
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
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
    def load(cls, path: str, task: str = "classification") -> "StockRandomForest":
        obj = cls.__new__(cls)
        obj.task = task
        obj.model = joblib.load(path)
        return obj
