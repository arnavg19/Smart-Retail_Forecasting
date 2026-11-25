from __future__ import annotations

import logging
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.data_loader import normalize_retail_columns
from src.features import add_feature_columns


LOGGER = logging.getLogger(__name__)


def compute_wape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    actual = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("WAPE requires equally sized, non-empty actual and predicted values.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("WAPE values must be finite.")
    denominator = np.sum(np.abs(actual))
    if denominator == 0:
        return 0.0 if np.allclose(actual, predicted) else float("nan")
    return float(np.sum(np.abs(actual - predicted)) / denominator)


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    item_column: str = "item_id",
    date_column: str = "invoice_date",
    target_column: str = "quantity",
    frequency: str = "D",
) -> pd.DataFrame:
    """Canonicalize headers, create a complete item-period grid, and add features."""
    data = df.copy()
    if not any(
        str(column).strip().lower() in {"unitprice", "unit_price", "price"}
        for column in data.columns
    ):
        data["unit_price"] = 0.0
    data = normalize_retail_columns(
        data, item_column, date_column, target_column, "unit_price"
    )
    try:
        frequency = pd.tseries.frequencies.to_offset(frequency).freqstr
    except ValueError as error:
        raise ValueError(f"Unsupported cross-validation frequency: {frequency!r}") from error
    data["item_id"] = data["item_id"].astype("string").str.strip()
    data["invoice_date"] = pd.to_datetime(data["invoice_date"], errors="coerce")
    data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce")
    data["unit_price"] = pd.to_numeric(data["unit_price"], errors="coerce")
    valid = (
        data["item_id"].notna()
        & data["item_id"].str.len().fillna(0).gt(0)
        & data["invoice_date"].notna()
        & np.isfinite(data["quantity"])
        & np.isfinite(data["unit_price"])
        & data["quantity"].ge(0)
        & data["unit_price"].ge(0)
    )
    invalid_rows = int((~valid).sum())
    if invalid_rows:
        LOGGER.warning("Excluded %s invalid rows while building features.", invalid_rows)
    data = data.loc[valid].copy()
    if data.empty:
        raise ValueError("No valid retail rows remain for feature engineering.")
    data["invoice_date"] = (
        data["invoice_date"].dt.to_period(frequency).dt.start_time
    )
    daily = (
        data.groupby(["item_id", "invoice_date"], as_index=False, sort=True)
        .agg(quantity=("quantity", "sum"), unit_price=("unit_price", "mean"))
    )
    calendar = pd.period_range(
        daily["invoice_date"].min(), daily["invoice_date"].max(), freq=frequency
    ).start_time
    grid = pd.MultiIndex.from_product(
        [daily["item_id"].drop_duplicates().sort_values(), calendar],
        names=["item_id", "invoice_date"],
    ).to_frame(index=False)
    complete = grid.merge(daily, on=["item_id", "invoice_date"], how="left")
    complete["quantity"] = complete["quantity"].fillna(0.0)
    complete["unit_price"] = (
        complete.groupby("item_id", sort=False)["unit_price"]
        .transform(lambda values: values.ffill().fillna(0.0))
        .fillna(0.0)
    )
    return add_feature_columns(complete, frequency=frequency)


def time_series_cv_evaluate(
    df: pd.DataFrame,
    target_col: str = "quantity",
    n_splits: int = 3,
    feature_cols: List[str] | None = None,
    *,
    date_col: str = "invoice_date",
    item_col: str = "item_id",
    frequency: str = "D",
) -> Dict[str, float]:
    """Evaluate expanding windows split on unique calendar dates, never panel rows."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least two.")
    try:
        frequency = pd.tseries.frequencies.to_offset(frequency).freqstr
    except ValueError as error:
        raise ValueError(f"Unsupported cross-validation frequency: {frequency!r}") from error
    data = build_feature_matrix(
        df,
        item_column=item_col,
        date_column=date_col,
        target_column=target_col,
        frequency=frequency,
    )
    if feature_cols is None:
        excluded = {"item_id", "invoice_date", "quantity", "gross_revenue"}
        feature_cols = [column for column in data.columns if column not in excluded]
    missing_features = sorted(set(feature_cols).difference(data.columns))
    if missing_features:
        raise ValueError(f"Unknown feature columns: {missing_features}")

    unique_dates = np.sort(data["invoice_date"].drop_duplicates().to_numpy())
    if unique_dates.size <= n_splits:
        raise ValueError(
            f"At least {n_splits + 1} distinct dates are required for {n_splits} folds."
        )
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict[str, float]] = []
    for train_idx, valid_idx in splitter.split(unique_dates):
        train_dates = unique_dates[train_idx]
        valid_dates = unique_dates[valid_idx]
        train = data.loc[data["invoice_date"].isin(train_dates)]
        valid = data.loc[data["invoice_date"].isin(valid_dates)]
        if train.empty or valid.empty:
            raise ValueError("Date-based cross-validation produced an empty fold.")
        x_train = train[feature_cols].copy()
        x_valid = valid[feature_cols].copy()
        if "item_id" in x_train:
            categories = pd.Index(x_train["item_id"].astype("string").unique())
            x_train["item_id"] = pd.Categorical(
                x_train["item_id"].astype("string"), categories=categories
            ).codes
            x_valid["item_id"] = pd.Categorical(
                x_valid["item_id"].astype("string"), categories=categories
            ).codes
        x_train = x_train.apply(pd.to_numeric, errors="raise").replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        x_valid = x_valid.apply(pd.to_numeric, errors="raise").replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.5,
            random_state=42,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=-1,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(x_train, train["quantity"])
        predictions = np.maximum(model.predict(x_valid), 0.0)
        baseline_column = "lag_7_days" if frequency == "D" else "lag_1_periods"
        baseline = valid[baseline_column].to_numpy(dtype=float)
        fold_metrics.append(
            {
                "WAPE": compute_wape(valid["quantity"], predictions),
                "RMSE": float(np.sqrt(mean_squared_error(valid["quantity"], predictions))),
                "MAE": float(mean_absolute_error(valid["quantity"], predictions)),
                "Baseline_WAPE": compute_wape(valid["quantity"], baseline),
                "Baseline_MAE": float(mean_absolute_error(valid["quantity"], baseline)),
            }
        )

    return {
        key: float(np.mean([metrics[key] for metrics in fold_metrics]))
        for key in fold_metrics[0]
    }


__all__ = ["compute_wape", "build_feature_matrix", "time_series_cv_evaluate"]
