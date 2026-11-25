from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Final

import numpy as np
import pandas as pd

from src.data_loader import normalize_retail_columns


LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


def add_cyclic_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add vectorized cyclic month and weekday features from ``invoice_date``.

    The input frame is copied, and its ``invoice_date`` column is converted to
    datetime when needed. Invalid dates raise an error rather than being
    silently discarded or encoded as missing values.
    """
    if "invoice_date" not in df.columns:
        raise ValueError("Input DataFrame must contain an 'invoice_date' column.")

    features = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(features["invoice_date"]):
        features["invoice_date"] = pd.to_datetime(features["invoice_date"], errors="raise")

    features["month"] = features["invoice_date"].dt.month
    features["day_of_week"] = features["invoice_date"].dt.dayofweek

    features["month_sin"] = np.sin(2.0 * np.pi * features["month"] / 12.0)
    features["month_cos"] = np.cos(2.0 * np.pi * features["month"] / 12.0)
    features["dow_sin"] = np.sin(2.0 * np.pi * features["day_of_week"] / 7.0)
    features["dow_cos"] = np.cos(2.0 * np.pi * features["day_of_week"] / 7.0)
    return features


def add_days_since_last_sale(df: pd.DataFrame) -> pd.DataFrame:
    """Add the elapsed days since the previous recorded date for each item.

    The input is sorted chronologically within each item before computing
    datetime differences. The first date for each item receives ``0.0``.

    Args:
        df: Input data containing ``item_id`` and ``invoice_date``.

    Returns:
        A sorted copy of ``df`` with a ``days_since_last_sale`` column.

    Raises:
        ValueError: If required columns are missing, item IDs or dates are
            missing, or dates cannot be converted to datetime.
    """
    required_columns = {"item_id", "invoice_date"}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_columns}")
    if df["item_id"].isna().any():
        raise ValueError("Input DataFrame must not contain null item_id values.")

    features = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(features["invoice_date"]):
        features["invoice_date"] = pd.to_datetime(features["invoice_date"], errors="raise")
    if features["invoice_date"].isna().any():
        raise ValueError("Input DataFrame must not contain missing invoice_date values.")
    features = features.sort_values(
        ["item_id", "invoice_date"], kind="mergesort"
    ).reset_index(drop=True)

    date_gaps = features.groupby("item_id", sort=False)["invoice_date"].diff()
    features["days_since_last_sale"] = (
        date_gaps.dt.total_seconds().div(86_400).fillna(0.0)
    )
    return features


def add_ewma_features(
    df: pd.DataFrame, span: int = 7, *, frequency: str = "D"
) -> pd.DataFrame:
    """Add a per-item, leakage-safe exponentially weighted quantity feature.

    Rows are sorted by ``item_id`` and ``invoice_date`` before calculation.
    The EWMA for each row uses only earlier quantities for that item, so the
    current target is not included. The initial unavailable value is filled
    with zero.

    Args:
        df: Input data containing ``item_id``, ``invoice_date``, and ``quantity``.
        span: EWMA span, measured in observations; defaults to seven.

    Returns:
        A sorted copy of ``df`` with an ``ewma_{span}d_qty`` column.

    Raises:
        ValueError: If required columns are missing, item IDs are null, or
            dates cannot be converted to datetime.
    """
    required_columns = {"item_id", "invoice_date", "quantity"}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_columns}")
    if span < 1:
        raise ValueError("span must be a positive integer.")
    if df["item_id"].isna().any():
        raise ValueError("Input DataFrame must not contain null item_id values.")

    features = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(features["invoice_date"]):
        features["invoice_date"] = pd.to_datetime(features["invoice_date"], errors="raise")
    features = features.sort_values(
        ["item_id", "invoice_date"], kind="mergesort"
    ).reset_index(drop=True)

    ewma = (
        features.groupby("item_id", sort=False)["quantity"]
        .ewm(span=span, adjust=False, min_periods=1)
        .mean()
        .groupby(level=0, sort=False)
        .shift(1)
    )
    period_label = "d" if frequency == "D" else "_periods"
    features[f"ewma_{span}{period_label}_qty"] = ewma.fillna(0.0).to_numpy()
    return features


def add_safe_discount_feature(
    df: pd.DataFrame, *, frequency: str = "D"
) -> pd.DataFrame:
    """Add a leakage-safe 30-observation price average and discount feature.

    Rows are sorted by item and date before calculating each item's prior
    30-observation rolling price mean. Discounts are computed only when the
    historical mean and current price are finite and the mean is positive;
    cold-start, zero-baseline, and invalid-price rows receive a zero discount.

    Args:
        df: Input data containing ``item_id``, ``invoice_date``, and
            ``unit_price``.

    Returns:
        A sorted copy of ``df`` with ``rolling_30d_avg_unit_price`` and
        ``discount_pct`` columns.

    Raises:
        ValueError: If required columns are missing, item IDs are null, or
            dates or prices cannot be converted to datetime or numeric values.
    """
    required_columns = {"item_id", "invoice_date", "unit_price"}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_columns}")
    if df["item_id"].isna().any():
        raise ValueError("Input DataFrame must not contain null item_id values.")

    features = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(features["invoice_date"]):
        features["invoice_date"] = pd.to_datetime(features["invoice_date"], errors="raise")
    if features["invoice_date"].isna().any():
        raise ValueError("Input DataFrame must not contain missing invoice_date values.")
    features["unit_price"] = pd.to_numeric(
        features["unit_price"], errors="raise"
    ).astype("float64")
    features = features.sort_values(
        ["item_id", "invoice_date"], kind="mergesort"
    ).reset_index(drop=True)

    rolling_price_mean = features.groupby("item_id", sort=False)["unit_price"].transform(
        lambda prices: prices.rolling(window=30, min_periods=1, closed="left").mean()
    )
    valid_mask = (
        rolling_price_mean.notna()
        & rolling_price_mean.gt(0.0)
        & np.isfinite(rolling_price_mean)
        & np.isfinite(features["unit_price"])
    )
    safe_denominator = rolling_price_mean.where(valid_mask, 1.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        discount_values = np.where(
            valid_mask.to_numpy(),
            ((rolling_price_mean - features["unit_price"]) / safe_denominator).to_numpy(),
            0.0,
        )

    period_label = "d" if frequency == "D" else "_periods"
    features[f"rolling_30{period_label}_avg_unit_price"] = (
        rolling_price_mean.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    features["discount_pct"] = np.nan_to_num(
        discount_values, copy=False, nan=0.0, posinf=0.0, neginf=0.0
    )
    return features


def add_price_discount_feature(
    df: pd.DataFrame, *, frequency: str = "D"
) -> pd.DataFrame:
    """Backward-compatible alias for :func:`add_safe_discount_feature`."""
    return add_safe_discount_feature(df, frequency=frequency)


def add_feature_columns(
    df: pd.DataFrame,
    *,
    item_column: str = "item_id",
    date_column: str = "invoice_date",
    target_column: str = "quantity",
    price_column: str = "unit_price",
    frequency: str = "D",
) -> pd.DataFrame:
    """Normalize common retail headers and create leakage-safe daily features."""
    data = df.copy()
    if price_column not in data.columns and not any(
        str(column).strip().lower() in {"unitprice", "unit_price", "price"}
        for column in data.columns
    ):
        data[price_column] = 0.0
    data = normalize_retail_columns(
        data, item_column, date_column, target_column, price_column
    )
    features = data.sort_values(["item_id", "invoice_date"], kind="mergesort").copy()
    features["invoice_date"] = pd.to_datetime(features["invoice_date"], errors="raise")
    if features["invoice_date"].isna().any() or features["item_id"].isna().any():
        raise ValueError("Feature data must have non-null item IDs and dates.")
    if features.duplicated(["item_id", "invoice_date"]).any():
        raise ValueError("Feature data must contain at most one row per item and date.")
    features["quantity"] = pd.to_numeric(features["quantity"], errors="raise")
    features["unit_price"] = pd.to_numeric(features["unit_price"], errors="raise")
    try:
        frequency = pd.tseries.frequencies.to_offset(frequency).freqstr
    except ValueError as error:
        raise ValueError(f"Unsupported feature frequency: {frequency!r}") from error
    lag_unit = "days" if frequency == "D" else "periods"
    rolling_unit = "d" if frequency == "D" else "_periods"
    features = add_cyclic_calendar_features(features)
    features = add_days_since_last_sale(features)
    for lag in (1, 7, 14, 30):
        features[f"lag_{lag}_{lag_unit}"] = (
            features.groupby("item_id", sort=False)["quantity"].shift(lag).fillna(0.0)
        )
    for window in (7, 30):
        prior = features.groupby("item_id", sort=False)["quantity"]
        features[f"rolling_{window}{rolling_unit}_mean_qty"] = prior.transform(
            lambda series: series.rolling(
                window=window, min_periods=1, closed="left"
            ).mean()
        ).fillna(0.0)
        features[f"rolling_{window}{rolling_unit}_std_qty"] = prior.transform(
            lambda series: series.rolling(
                window=window, min_periods=1, closed="left"
            ).std(ddof=0)
        ).fillna(0.0)
    return features


class RetailFeatureEngineer:
    """Build retail demand features for a time-series forecasting pipeline."""

    REQUIRED_COLUMNS: Final[set[str]] = {"item_id", "invoice_date", "quantity", "unit_price"}
    LAG_WINDOWS: Final[list[int]] = [1, 7, 14, 30]
    ROLLING_WINDOWS: Final[list[int]] = [7, 30]

    def __init__(
        self,
        data_path: str | Path,
        *,
        item_column: str = "item_id",
        date_column: str = "invoice_date",
        target_column: str = "quantity",
        price_column: str = "unit_price",
        frequency: str = "D",
    ) -> None:
        self.data_path = Path(data_path)
        self.item_column = item_column
        self.date_column = date_column
        self.target_column = target_column
        self.price_column = price_column
        try:
            self.frequency = pd.tseries.frequencies.to_offset(frequency).freqstr
        except ValueError as error:
            raise ValueError(f"Unsupported feature frequency: {frequency!r}") from error
        self.lag_unit = "days" if self.frequency == "D" else "periods"
        self.rolling_unit = "d" if self.frequency == "D" else "_periods"
        self.ewma_column = f"ewma_7{self.rolling_unit}_qty"
        self.price_average_column = f"rolling_30{self.rolling_unit}_avg_unit_price"
        self.data: pd.DataFrame | None = None

    def load_data(self) -> pd.DataFrame:
        """Load item-period data and construct a complete configured-frequency grid.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Daily demand data not found at: {self.data_path}")

        df = pd.read_csv(
            self.data_path,
            dtype={
                "StockCode": "string",
                "stockcode": "string",
                "stock_code": "string",
                "item_id": "string",
                self.item_column: "string",
            },
            low_memory=False,
        )
        df = normalize_retail_columns(
            df,
            self.item_column,
            self.date_column,
            self.target_column,
            self.price_column,
        )
        missing = sorted(self.REQUIRED_COLUMNS.difference(df.columns))
        if missing:
            raise ValueError(f"Input dataset is missing required columns: {missing}")

        self.data = df[["item_id", "invoice_date", "quantity", "unit_price"]].copy()
        self.data["invoice_date"] = pd.to_datetime(self.data["invoice_date"], errors="coerce")
        self.data["quantity"] = pd.to_numeric(self.data["quantity"], errors="coerce")
        self.data["unit_price"] = pd.to_numeric(self.data["unit_price"], errors="coerce")
        self.data = self.data.dropna(
            subset=["item_id", "invoice_date", "quantity", "unit_price"]
        ).copy()
        if self.data.empty:
            raise ValueError("No valid item-period rows remain after feature-data validation.")
        try:
            self.data["invoice_date"] = (
                self.data["invoice_date"].dt.to_period(self.frequency).dt.start_time
            )
        except ValueError as error:
            raise ValueError(f"Unsupported feature frequency: {self.frequency!r}") from error
        self.data = (
            self.data.groupby(["item_id", "invoice_date"], as_index=False, sort=True)
            .agg(quantity=("quantity", "sum"), unit_price=("unit_price", "mean"))
        )
        calendar = pd.period_range(
            self.data["invoice_date"].min(),
            self.data["invoice_date"].max(),
            freq=self.frequency,
        ).start_time
        grid = pd.MultiIndex.from_product(
            [self.data["item_id"].drop_duplicates().sort_values(), calendar],
            names=["item_id", "invoice_date"],
        ).to_frame(index=False)
        self.data = grid.merge(
            self.data, on=["item_id", "invoice_date"], how="left"
        )
        self.data["quantity"] = self.data["quantity"].fillna(0.0)
        self.data["unit_price"] = (
            self.data.groupby("item_id", sort=False)["unit_price"]
            .transform(lambda prices: prices.ffill().fillna(0.0))
            .fillna(0.0)
        )
        return self.data

    def create_temporal_features(self) -> pd.DataFrame:
        """Create calendar-driven demand signals for recurring retail seasonality.

        Calendar variables matter in retail because demand often shifts by weekday, month, and consumer
        pay-cycle patterns. These features describe when an item-day occurred without leaking future
        information from other observations.
        """
        if self.data is None:
            self.load_data()
        features = add_days_since_last_sale(
            add_cyclic_calendar_features(self.data)
        )

        features["day_of_month"] = features["invoice_date"].dt.day
        features["quarter"] = features["invoice_date"].dt.quarter
        features["is_weekend"] = features["day_of_week"].isin([5, 6]).astype(int)
        features["payday_flag"] = features["invoice_date"].dt.day.between(1, 5).astype(int)

        return features

    def create_lag_features(self) -> pd.DataFrame:
        """Create lagged demand features for short- and medium-horizon item demand dependence.

        These lag signals encode how sales on previous days influence current demand, especially for
        promotions, replenishment cycles, and weekday purchasing behavior. Lagged quantities are built
        per item and intentionally use only historical information to avoid temporal leakage.
        """
        if self.data is None:
            self.load_data()

        features = self.data.copy().sort_values(["item_id", "invoice_date"]).reset_index(drop=True)
        for lag in self.LAG_WINDOWS:
            features[f"lag_{lag}_{self.lag_unit}"] = (
                features.groupby("item_id")["quantity"]
                .transform(lambda s: s.shift(lag))
                .fillna(0.0)
            )
        return features

    def create_rolling_features(self) -> pd.DataFrame:
        """Compute rolling demand statistics and relative price perception features.

        Rolling statistics summarize recent demand volatility and trend while limiting each observation to
        the historical window ending at the current day. Setting closed='left' ensures the current day is
        not included in the rolling window, so the model never sees the same-day target as a feature.
        """
        if self.data is None:
            self.load_data()

        features = add_safe_discount_feature(self.data, frequency=self.frequency)

        grouped = features.groupby("item_id", sort=True)
        for window in self.ROLLING_WINDOWS:
            rolling_stats = grouped["quantity"].transform(
                lambda s: s.rolling(window=window, min_periods=1, closed="left").mean()
            )
            rolling_std = grouped["quantity"].transform(
                lambda s: s.rolling(window=window, min_periods=1, closed="left").std(ddof=0)
            )
            rolling_min = grouped["quantity"].transform(
                lambda s: s.rolling(window=window, min_periods=1, closed="left").min()
            )
            rolling_max = grouped["quantity"].transform(
                lambda s: s.rolling(window=window, min_periods=1, closed="left").max()
            )

            features[f"rolling_{window}{self.rolling_unit}_mean_qty"] = rolling_stats.fillna(0.0)
            features[f"rolling_{window}{self.rolling_unit}_std_qty"] = rolling_std.fillna(0.0)
            features[f"rolling_{window}{self.rolling_unit}_min_qty"] = rolling_min.fillna(0.0)
            features[f"rolling_{window}{self.rolling_unit}_max_qty"] = rolling_max.fillna(0.0)

        features["relative_price_ratio"] = (
            features["unit_price"] / features[self.price_average_column].replace(0, pd.NA)
        ).fillna(1.0)

        return features

    def transform(self) -> pd.DataFrame:
        """Assemble the final feature-rich dataset ready for supervised forecasting.

        The transformation pipeline combines temporal, lag, rolling demand, and pricing perception signals
        while dropping warm-up rows created by lag and rolling calculations. This yields a clean feature set
        without contaminating historical training windows with unavailable prior data.
        """
        if self.data is None:
            self.load_data()

        started_at = perf_counter()
        combined = self.create_temporal_features()
        base_columns = self.data.columns
        lag_features = self.create_lag_features()
        rolling_features = self.create_rolling_features()
        ewma_features = add_ewma_features(self.data, frequency=self.frequency)
        combined = combined.join(lag_features.drop(columns=base_columns, errors="ignore"))
        combined = combined.join(rolling_features.drop(columns=base_columns, errors="ignore"))
        combined = combined.join(ewma_features[[self.ewma_column]])

        # Reuse the sorted daily dataset and remove warm-up rows created by lag/rolling windows.
        max_lag = max(self.LAG_WINDOWS)
        max_window = max(self.ROLLING_WINDOWS)
        warmup_periods = max(max_lag, max_window)

        combined = combined.sort_values(["item_id", "invoice_date"]).reset_index(drop=True)
        combined["item_day_index"] = combined.groupby("item_id").cumcount()
        combined = combined[combined["item_day_index"] >= warmup_periods].copy()
        combined = combined.drop(columns=["item_day_index"])

        output_cols = [
            "item_id",
            "invoice_date",
            "quantity",
            "unit_price",
            "day_of_week",
            "day_of_month",
            "days_since_last_sale",
            "month",
            "quarter",
            "month_sin",
            "month_cos",
            "dow_sin",
            "dow_cos",
            "is_weekend",
            "payday_flag",
            self.ewma_column,
            "discount_pct",
            *[f"lag_{lag}_{self.lag_unit}" for lag in self.LAG_WINDOWS],
            *[f"rolling_{window}{self.rolling_unit}_mean_qty" for window in self.ROLLING_WINDOWS],
            *[f"rolling_{window}{self.rolling_unit}_std_qty" for window in self.ROLLING_WINDOWS],
            *[f"rolling_{window}{self.rolling_unit}_min_qty" for window in self.ROLLING_WINDOWS],
            *[f"rolling_{window}{self.rolling_unit}_max_qty" for window in self.ROLLING_WINDOWS],
            self.price_average_column,
            "relative_price_ratio",
        ]

        available = [column for column in output_cols if column in combined.columns]
        transformed = combined[available].copy()
        transformed = transformed.dropna().reset_index(drop=True)

        output_path = self.data_path.parent / "daily_demand_features.csv"
        transformed.to_csv(output_path, index=False)
        LOGGER.info(
            "Feature transformation saved %s rows to %s in %.3f seconds",
            len(transformed),
            output_path,
            perf_counter() - started_at,
        )
        return transformed


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    source_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "daily_demand_clean.csv"
    engineer = RetailFeatureEngineer(source_path)
    features = engineer.transform()
    LOGGER.info("Saved %s feature rows to %s", len(features), source_path.parent / "daily_demand_features.csv")
