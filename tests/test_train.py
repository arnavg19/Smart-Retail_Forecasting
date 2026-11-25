import json

import numpy as np
import pandas as pd
import pytest

import src.predict as predict_module
from src.train import (
    TrainingConfig,
    chronological_split,
    rmsle,
    train_model,
    wape,
)
from src.evaluate_features import evaluate_feature_importance
from src.predict import predict_test_set


def test_rmsle_clips_negative_predictions():
    assert rmsle([0.0, 3.0], [-1.0, 3.0]) == pytest.approx(
        np.sqrt((0.0**2 + (np.log1p(3.0) - np.log1p(3.0)) ** 2) / 2)
    )


def test_wape_handles_zero_total_actuals():
    assert np.isnan(wape([0.0, 0.0], [1.0, 1.0]))
    assert wape([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert wape([2.0, 2.0], [1.0, 3.0]) == pytest.approx(0.5)


def test_chronological_split_separates_validation_dates():
    data = pd.DataFrame(
        {
            "invoice_date": pd.to_datetime(
                ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-03"]
            ),
            "quantity": [3, 1, 2, 4],
        }
    )

    train, validation = chronological_split(data, validation_fraction=0.34)

    assert train["invoice_date"].max() < validation["invoice_date"].min()
    assert validation["invoice_date"].min() == pd.Timestamp("2024-01-03")


def test_chronological_split_keeps_same_calendar_date_together():
    data = pd.DataFrame(
        {
            "invoice_date": pd.to_datetime(
                [
                    "2024-01-01 08:00",
                    "2024-01-01 18:00",
                    "2024-01-02 09:00",
                    "2024-01-02 17:00",
                ]
            ),
            "quantity": [1, 2, 3, 4],
        }
    )

    train, validation = chronological_split(data, validation_fraction=0.5)

    assert train["invoice_date"].dt.date.nunique() == 1
    assert validation["invoice_date"].dt.date.nunique() == 1
    assert train["invoice_date"].max() < validation["invoice_date"].min()


def test_train_model_saves_model_and_metadata(tmp_path, monkeypatch):
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    data = pd.DataFrame(
        [
            {
                "item_id": item_id,
                "invoice_date": date,
                "quantity": float((day_index + item_index) % 4),
                "day_of_week": date.dayofweek,
                "unit_price": (
                    float("inf")
                    if day_index == 0
                    else float("nan")
                    if day_index == 1
                    else 5.0 + item_index
                ),
            }
            for item_index, item_id in enumerate(["A", "B"])
            for day_index, date in enumerate(dates)
        ]
    )
    data_path = tmp_path / "features.csv"
    data.to_csv(data_path, index=False)
    models_dir = tmp_path / "models"

    metadata = train_model(
        TrainingConfig(
            data_path=data_path,
            models_dir=models_dir,
            validation_fraction=0.2,
            n_estimators=6,
            early_stopping_rounds=2,
            n_jobs=1,
        )
    )

    model_path = models_dir / "lightgbm_tweedie_model.joblib"
    metadata_path = models_dir / "lightgbm_tweedie_metadata.json"
    assert model_path.is_file()
    assert metadata_path.is_file()
    assert metadata["objective"] == "tweedie"
    assert metadata["tweedie_variance_power"] == 1.5
    assert metadata["categorical_features"] == ["item_id"]
    assert metadata["validation_start_date"] > metadata["train_end_date"]
    assert metadata["metrics"]["validation_baseline_wape"] is not None
    assert metadata["metrics"]["validation_baseline_rmsle"] is not None
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["train_rows"] == 64

    importance_path = models_dir / "feature_importance.png"
    importance = evaluate_feature_importance(model_path, importance_path)
    assert not importance.empty
    assert importance_path.is_file()

    test_data = pd.DataFrame(
        [
            {
                "item_id": item_id,
                "invoice_date": date,
                "quantity": float(index % 3),
                "day_of_week": date.dayofweek,
                "unit_price": 7.0,
            }
            for index, (item_id, date) in enumerate(
                [
                    ("A", pd.Timestamp("2024-02-10")),
                    ("B", pd.Timestamp("2024-02-10")),
                    ("NEW", pd.Timestamp("2024-02-10")),
                    ("A", pd.Timestamp("2024-02-11")),
                ]
            )
        ]
    )
    test_path = tmp_path / "test_features.parquet"
    test_data.to_parquet(test_path, index=False)
    predictions_path = models_dir / "test_predictions.csv"

    metrics, predictions = predict_test_set(
        model_path=model_path,
        metadata_path=metadata_path,
        data_path=test_path,
        output_path=predictions_path,
    )

    assert np.isfinite(metrics["rmsle"])
    assert np.isfinite(metrics["wape"])
    assert metrics["evaluation_split"] == "test_set"
    assert (predictions["predicted_quantity"] >= 0.0).all()
    assert list(predictions.columns) == [
        "invoice_date",
        "stock_code",
        "actual_quantity",
        "predicted_quantity",
        "evaluation_split",
    ]
    assert predictions["evaluation_split"].unique().tolist() == ["test_set"]
    assert predictions_path.is_file()

    fallback_path = tmp_path / "daily_demand_features.csv"
    data.to_csv(fallback_path, index=False)
    missing_parquet_path = tmp_path / "missing_default_test_features.parquet"
    fallback_output_path = models_dir / "fallback_predictions.csv"
    monkeypatch.setattr(predict_module, "DEFAULT_DATA_PATH", missing_parquet_path)
    monkeypatch.setattr(predict_module, "DEFAULT_FALLBACK_DATA_PATH", fallback_path)

    fallback_metrics, fallback_predictions = predict_test_set(
        model_path=model_path,
        metadata_path=metadata_path,
        data_path=missing_parquet_path,
        output_path=fallback_output_path,
    )

    assert len(fallback_predictions) == 16
    assert fallback_metrics["evaluation_split"] == "validation_holdout"
    assert fallback_predictions["evaluation_split"].unique().tolist() == [
        "validation_holdout"
    ]
    assert fallback_predictions["invoice_date"].min() >= pd.Timestamp(
        metadata["validation_start_date"]
    )
    assert fallback_output_path.is_file()
