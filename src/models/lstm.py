import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

try:
    import tensorflow as tf  # noqa: F401
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


def build_sequences(data: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, 0])  # first column is the target (close price)
    return np.array(X), np.array(y)


class StockLSTM:
    def __init__(
        self,
        seq_len: int = 60,
        units: list[int] = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        task: str = "regression",
    ):
        self.seq_len = seq_len
        self.units = units or [128, 64]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.task = task
        self.scaler = MinMaxScaler()
        self.model = None
        self.history = None

    def _build_model(self, n_features: int):
        if not _TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM. Install it with: pip install tensorflow")
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

        model = Sequential()
        for i, units in enumerate(self.units):
            return_seq = i < len(self.units) - 1
            if i == 0:
                model.add(LSTM(units, return_sequences=return_seq, input_shape=(self.seq_len, n_features)))
            else:
                model.add(LSTM(units, return_sequences=return_seq))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout))

        if self.task == "classification":
            model.add(Dense(1, activation="sigmoid"))
            model.compile(
                optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
        else:
            model.add(Dense(1))
            model.compile(
                optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                loss="huber",
                metrics=["mae"],
            )

        self.model = model
        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> "StockLSTM":
        import tensorflow as tf

        n_features = X_train.shape[2]
        self._build_model(n_features)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
        ]

        val_data = (X_val, y_val) if X_val is not None else None
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0).flatten()

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "close",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cols = [target_col] + [c for c in feature_cols if c != target_col]
        data = df[cols].values
        data_scaled = self.scaler.fit_transform(data)
        X, y = build_sequences(data_scaled, self.seq_len)
        return X, y, data_scaled

    def inverse_transform_predictions(self, preds: np.ndarray) -> np.ndarray:
        dummy = np.zeros((len(preds), self.scaler.n_features_in_))
        dummy[:, 0] = preds
        return self.scaler.inverse_transform(dummy)[:, 0]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load_weights(self, path: str) -> None:
        import tensorflow as tf
        self.model = tf.keras.models.load_model(path)
