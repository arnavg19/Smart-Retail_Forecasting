from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.features import add_feature_columns


def compute_wape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return 0.0
    return np.sum(np.abs(y_true - y_pred)) / denom


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare cleaned and engineered features for demand modeling."""
    return add_feature_columns(df)


def time_series_cv_evaluate(
    df: pd.DataFrame,
    target_col: str = "Quantity",
    n_splits: int = 3,
    feature_cols: List[str] | None = None,
) -> Dict[str, float]:
    """Run expanding-window cross-validation and return key metrics."""
    data = build_feature_matrix(df).copy()
    if feature_cols is None:
        feature_cols = [
            c for c in data.columns
            if c not in {"StockCode", "InvoiceDate", target_col, "GrossRevenue"}
        ]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for train_idx, valid_idx in tscv.split(data):
        train = data.iloc[train_idx]
        valid = data.iloc[valid_idx]

        X_train = train[feature_cols]
        y_train = train[target_col]
        X_valid = valid[feature_cols]
        y_valid = valid[target_col]

        model = None
        try:
            from lightgbm import LGBMRegressor

            model = LGBMRegressor(
                objective="regression_tweedie",
                random_state=42,
                n_estimators=200,
                learning_rate=0.05,
                max_depth=-1,
                n_jobs=-1,
            )
        except Exception:
            from sklearn.dummy import DummyRegressor

            model = DummyRegressor(strategy="mean")

        model.fit(X_train, y_train)
        predictions = model.predict(X_valid)
        fold_metrics.append(
            {
                "WAPE": compute_wape(y_valid, predictions),
                "RMSE": float(np.sqrt(mean_squared_error(y_valid, predictions))),
                "MAE": float(mean_absolute_error(y_valid, predictions)),
            }
        )

    if not fold_metrics:
        return {"WAPE": 0.0, "RMSE": 0.0, "MAE": 0.0}

    return {
        "WAPE": float(np.mean([m["WAPE"] for m in fold_metrics])),
        "RMSE": float(np.mean([m["RMSE"] for m in fold_metrics])),
        "MAE": float(np.mean([m["MAE"] for m in fold_metrics])),
    }


__all__ = ["compute_wape", "build_feature_matrix", "time_series_cv_evaluate"]
