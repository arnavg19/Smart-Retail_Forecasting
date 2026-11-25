import numpy as np
import pandas as pd

import lightgbm

import src.forecasting as forecasting


def test_feature_matrix_normalizes_headers_and_expands_missing_dates():
    raw = pd.DataFrame(
        {
            "StockCode": ["A", "A", "B", "B"],
            "InvoiceDate": [
                "2024-01-01",
                "2024-01-03",
                "2024-01-01",
                "2024-01-03",
            ],
            "Quantity": [2, 4, 1, 3],
            "UnitPrice": [10, 12, 8, 9],
        }
    )

    features = forecasting.build_feature_matrix(raw)

    assert len(features) == 6
    assert set(features["item_id"]) == {"A", "B"}
    assert features.loc[
        features["invoice_date"].eq(pd.Timestamp("2024-01-02")), "quantity"
    ].eq(0).all()


def test_cross_validation_keeps_each_calendar_date_in_one_fold(monkeypatch):
    rows = []
    for day_index, date in enumerate(pd.date_range("2024-01-01", periods=6)):
        for item_id in ("A", "B"):
            rows.append(
                {
                    "item_id": item_id,
                    "invoice_date": date,
                    "quantity": float(day_index + 1),
                    "lag_7_days": 0.0,
                    "date_index": float(day_index),
                }
            )
    engineered = pd.DataFrame(rows)
    monkeypatch.setattr(forecasting, "build_feature_matrix", lambda *args, **kwargs: engineered)

    folds = []

    class FakeRegressor:
        def __init__(self, **configuration):
            self.configuration = configuration

        def fit(self, features, target):
            self.train_dates = features["date_index"].to_numpy()
            self.train_target_size = len(target)
            return self

        def predict(self, features):
            valid_dates = features["date_index"].to_numpy()
            folds.append((self.train_dates.copy(), valid_dates.copy()))
            return np.zeros(len(features))

    monkeypatch.setattr(lightgbm, "LGBMRegressor", FakeRegressor)

    result = forecasting.time_series_cv_evaluate(pd.DataFrame(), n_splits=2)

    assert len(folds) == 2
    for train_dates, valid_dates in folds:
        assert train_dates.max() < valid_dates.min()
        assert len(train_dates) % 2 == 0
        assert len(valid_dates) % 2 == 0
    assert "Baseline_WAPE" in result
