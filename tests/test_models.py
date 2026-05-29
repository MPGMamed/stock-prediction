import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from src.models.random_forest import StockRandomForest
from src.models.xgboost_model import StockXGBoost
from src.evaluation.metrics import (
    classification_metrics, regression_metrics, sharpe_ratio, max_drawdown,
)


def clf_data():
    X, y = make_classification(n_samples=300, n_features=20, random_state=42)
    return X[:200], y[:200], X[200:], y[200:]


def reg_data():
    X, y = make_regression(n_samples=300, n_features=20, noise=0.1, random_state=42)
    return X[:200], y[:200], X[200:], y[200:]


class TestRandomForest:
    def test_classification_fit_predict(self):
        Xtr, ytr, Xte, yte = clf_data()
        model = StockRandomForest(task="classification", n_estimators=50)
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        assert len(preds) == len(yte)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shape(self):
        Xtr, ytr, Xte, yte = clf_data()
        model = StockRandomForest(task="classification", n_estimators=50)
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)
        assert proba.shape == (len(yte), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_feature_importance_length(self):
        Xtr, ytr, _, _ = clf_data()
        model = StockRandomForest(task="classification", n_estimators=50)
        model.fit(Xtr, ytr)
        names = [f"f{i}" for i in range(Xtr.shape[1])]
        fi = model.feature_importance(names)
        assert len(fi) == Xtr.shape[1]

    def test_save_load(self, tmp_path):
        Xtr, ytr, Xte, yte = clf_data()
        model = StockRandomForest(task="classification", n_estimators=50)
        model.fit(Xtr, ytr)
        path = str(tmp_path / "rf.pkl")
        model.save(path)
        loaded = StockRandomForest.load(path, task="classification")
        np.testing.assert_array_equal(model.predict(Xte), loaded.predict(Xte))


class TestXGBoost:
    def test_classification_accuracy(self):
        Xtr, ytr, Xte, yte = clf_data()
        model = StockXGBoost(task="classification", n_estimators=50)
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        metrics = classification_metrics(yte, preds)
        assert metrics["accuracy"] > 0.5

    def test_regression_rmse(self):
        Xtr, ytr, Xte, yte = reg_data()
        model = StockXGBoost(task="regression", n_estimators=50)
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        metrics = regression_metrics(yte, preds)
        assert metrics["rmse"] < yte.std() * 2


class TestMetrics:
    def test_sharpe_positive(self):
        returns = np.random.randn(252) * 0.01 + 0.001
        assert sharpe_ratio(returns) > 0

    def test_max_drawdown_negative(self):
        equity = np.cumprod(1 + np.random.randn(100) * 0.01)
        dd = max_drawdown(equity)
        assert dd <= 0

    def test_classification_metrics_keys(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        metrics = classification_metrics(y_true, y_pred)
        assert {"accuracy", "precision", "recall", "f1"}.issubset(metrics.keys())
