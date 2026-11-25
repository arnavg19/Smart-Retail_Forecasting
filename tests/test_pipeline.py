import pandas as pd

import pandas as pd
import pytest

from src.data_loader import (
    clean_transactions,
    clean_transactions_with_audit,
    expand_daily_sales_grid,
)
from src.features import (
    RetailFeatureEngineer,
    add_cyclic_calendar_features,
    add_days_since_last_sale,
    add_ewma_features,
    add_safe_discount_feature,
)
from src.pricing_engine import demand_curve, optimize_price


@pytest.fixture
def engineered_output(tmp_path):
    source = pd.DataFrame(
        {
            "item_id": ["A1"] * 40,
            "invoice_date": pd.date_range("2024-01-01", periods=40, freq="D"),
            "quantity": list(range(1, 41)),
            "unit_price": [12.0] * 40,
        }
    )
    data_path = tmp_path / "daily_demand_clean.csv"
    source.to_csv(data_path, index=False)
    features = RetailFeatureEngineer(data_path).transform()
    return source, features


def test_clean_transactions_removes_invalid_rows():
    df = pd.DataFrame(
        {
            "item_id": ["A1", "A1", "A2", "A3", "A4", "A5"],
            "quantity": [10, -2, 0, 5, 8, float("inf")],
            "unit_price": [12.0, 3.0, 5.0, -1.0, 25.0, 4.0],
            "invoice_date": pd.to_datetime([
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-06",
            ]),
        }
    )

    cleaned = clean_transactions(df)

    assert list(cleaned["item_id"]) == ["A1", "A4"]
    assert cleaned["quantity"].min() > 0
    assert cleaned["unit_price"].min() > 0


def test_expand_daily_sales_grid_fills_zero_sales():
    df = pd.DataFrame(
        {
            "item_id": ["A1", "A1", "A2"],
            "quantity": [2, 5, 3],
            "unit_price": [10.0, 10.0, 11.0],
            "invoice_date": pd.to_datetime([
                "2024-01-01",
                "2024-01-03",
                "2024-01-02",
            ]),
        }
    )

    expanded = expand_daily_sales_grid(
        df,
        min_sales=1,
        date_start="2024-01-01",
        date_end="2024-01-03",
    )

    assert set(expanded["item_id"]) == {"A1", "A2"}
    assert len(expanded) == 6
    assert (expanded["quantity"] == 0).any()


def test_feature_matrix_has_no_nans_in_key_lag_columns(engineered_output):
    _, feature_matrix = engineered_output
    assert {"lag_1_days", "rolling_7d_mean_qty"}.issubset(feature_matrix.columns)

    assert not feature_matrix[["lag_1_days", "rolling_7d_mean_qty"]].isna().any().any()


def test_expanded_grid_unit_price_is_positive_after_filling():
    df = pd.DataFrame(
        {
            "item_id": ["A1", "A1"],
            "invoice_date": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "quantity": [2, 5],
            "unit_price": [10.0, 12.0],
        }
    )

    expanded = expand_daily_sales_grid(
        df,
        min_sales=1,
        date_start="2024-01-01",
        date_end="2024-01-05",
    )

    assert expanded["unit_price"].notna().all()
    assert expanded["unit_price"].tolist() == [0.0, 10.0, 10.0, 12.0, 12.0]


def test_lag_one_uses_previous_day_quantity(engineered_output):
    source, feature_matrix = engineered_output
    assert "lag_1_days" in feature_matrix.columns
    sample = feature_matrix.iloc[0]
    previous_date = sample["invoice_date"] - pd.Timedelta(days=1)
    previous_quantity = source.loc[
        source["invoice_date"] == previous_date,
        "quantity",
    ].iloc[0]

    assert sample["lag_1_days"] == previous_quantity
    assert sample["lag_1_days"] != sample["quantity"]


def test_feature_matrix_contains_expected_output_schema(engineered_output):
    _, feature_matrix = engineered_output
    expected_columns = {
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
        "ewma_7d_qty",
        "discount_pct",
        "lag_1_days",
        "lag_7_days",
        "lag_14_days",
        "lag_30_days",
        "rolling_7d_mean_qty",
        "rolling_30d_mean_qty",
        "rolling_7d_std_qty",
        "rolling_30d_std_qty",
        "rolling_7d_min_qty",
        "rolling_30d_min_qty",
        "rolling_7d_max_qty",
        "rolling_30d_max_qty",
        "rolling_30d_avg_unit_price",
        "relative_price_ratio",
    }

    assert expected_columns.issubset(feature_matrix.columns)


def test_feature_engineer_supports_configured_columns_and_weekly_periods(tmp_path):
    source = pd.DataFrame(
        {
            "SKU": ["A"] * 280,
            "Sold On": pd.date_range("2024-01-01", periods=280, freq="D"),
            "Units": [1.0] * 280,
            "Price USD": [10.0] * 280,
        }
    )
    path = tmp_path / "weekly_source.csv"
    source.to_csv(path, index=False)

    features = RetailFeatureEngineer(
        path,
        item_column="SKU",
        date_column="Sold On",
        target_column="Units",
        price_column="Price USD",
        frequency="W",
    ).transform()

    assert "lag_1_periods" in features.columns
    assert "rolling_7_periods_mean_qty" in features.columns
    assert "ewma_7_periods_qty" in features.columns
    assert features["invoice_date"].dt.dayofweek.eq(0).all()


def test_add_cyclic_calendar_features_converts_dates_and_encodes_calendar():
    source = pd.DataFrame({"invoice_date": ["2023-12-31", "2024-01-01"]})

    features = add_cyclic_calendar_features(source)

    assert pd.api.types.is_datetime64_any_dtype(features["invoice_date"])
    assert features["month"].tolist() == [12, 1]
    assert features["day_of_week"].tolist() == [6, 0]
    assert features["month_sin"].tolist() == pytest.approx([0.0, 0.5], abs=1e-12)
    assert features["month_cos"].tolist() == pytest.approx([1.0, 3**0.5 / 2], abs=1e-12)
    assert features["dow_sin"].iloc[1] == pytest.approx(0.0, abs=1e-12)
    assert features["dow_cos"].iloc[1] == pytest.approx(1.0)
    assert source["invoice_date"].tolist() == ["2023-12-31", "2024-01-01"]


def test_add_days_since_last_sale_sorts_and_calculates_gaps_per_item():
    source = pd.DataFrame(
        {
            "item_id": ["B", "A", "B", "A"],
            "invoice_date": [
                "2024-01-05",
                "2024-01-10",
                "2024-01-01",
                "2024-01-01",
            ],
        }
    )

    features = add_days_since_last_sale(source)

    assert features[["item_id", "invoice_date"]].values.tolist() == [
        ["A", pd.Timestamp("2024-01-01")],
        ["A", pd.Timestamp("2024-01-10")],
        ["B", pd.Timestamp("2024-01-01")],
        ["B", pd.Timestamp("2024-01-05")],
    ]
    assert features["days_since_last_sale"].tolist() == pytest.approx(
        [0.0, 9.0, 0.0, 4.0]
    )


def test_add_ewma_features_sorts_groups_and_uses_only_prior_quantities():
    source = pd.DataFrame(
        {
            "item_id": ["B", "A", "B", "A"],
            "invoice_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-01", "2024-01-01"]
            ),
            "quantity": [20.0, 2.0, 10.0, 1.0],
        }
    )

    features = add_ewma_features(source)

    assert features[["item_id", "invoice_date"]].values.tolist() == [
        ["A", pd.Timestamp("2024-01-01")],
        ["A", pd.Timestamp("2024-01-02")],
        ["B", pd.Timestamp("2024-01-01")],
        ["B", pd.Timestamp("2024-01-02")],
    ]
    assert features["ewma_7d_qty"].tolist() == pytest.approx([0.0, 1.0, 0.0, 10.0])


def test_add_price_discount_feature_uses_only_prior_item_prices():
    source = pd.DataFrame(
        {
            "item_id": ["B", "A", "A", "B", "A"],
            "invoice_date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-03",
                ]
            ),
            "unit_price": [0.0, 8.0, 10.0, 0.0, 12.0],
        }
    )

    features = add_safe_discount_feature(source)

    assert features["item_id"].tolist() == ["A", "A", "A", "B", "B"]
    assert features["rolling_30d_avg_unit_price"].tolist() == pytest.approx(
        [0.0, 10.0, 9.0, 0.0, 0.0]
    )
    assert features["discount_pct"].tolist() == pytest.approx(
        [0.0, 0.2, -1.0 / 3.0, 0.0, 0.0]
    )


def test_add_price_discount_feature_fills_nan_and_infinite_values():
    source = pd.DataFrame(
        {
            "item_id": ["A1"] * 4,
            "invoice_date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "unit_price": [10.0, float("inf"), float("nan"), 5.0],
        }
    )

    features = add_safe_discount_feature(source)

    assert features["rolling_30d_avg_unit_price"].notna().all()
    assert features["discount_pct"].notna().all()
    assert not features["discount_pct"].isin([float("inf"), float("-inf")]).any()


def test_optimize_price_respects_guardrails():
    result = optimize_price(
        predicted_demand=120.0,
        unit_cost=20.0,
        inventory=200,
        historical_max_price=100.0,
        reference_price=50.0,
        elasticity=-1.5,
    )

    assert result["optimal_price"] >= 23.0
    assert result["optimal_price"] <= 120.0
    assert result["expected_profit"] >= 0


def test_clean_transactions_accepts_configured_source_headers_and_reports_counts():
    raw = pd.DataFrame(
        {
            "SKU": ["A", "A", None],
            "Sold On": ["2024-01-01", "2024-01-03", "invalid"],
            "Units": [2, 5, 1],
            "Price USD": [10.0, 12.0, 1.0],
        }
    )

    cleaned, audit = clean_transactions_with_audit(
        raw,
        item_column="SKU",
        date_column="Sold On",
        target_column="Units",
        price_column="Price USD",
    )

    assert cleaned["item_id"].tolist() == ["A", "A"]
    assert audit == {
        "input_rows": 3,
        "valid_transaction_rows": 2,
        "dropped_invalid_rows": 1,
        "active_skus": 1,
    }


def test_expand_grid_supports_weekly_frequency():
    source = pd.DataFrame(
        {
            "StockCode": ["A", "A"],
            "InvoiceDate": ["2024-01-02", "2024-01-10"],
            "Quantity": [2, 3],
            "UnitPrice": [10.0, 12.0],
        }
    )

    expanded = expand_daily_sales_grid(
        source, min_sales=1, frequency="W", date_start="2024-01-02", date_end="2024-01-10"
    )

    assert expanded["invoice_date"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-08"),
    ]
    assert expanded["quantity"].tolist() == [2.0, 3.0]


def test_pricing_rejects_invalid_inputs_and_respects_inventory_and_history():
    with pytest.raises(ValueError, match="unit_cost"):
        optimize_price(10, 0, 5, 20, 10, 1.0)
    with pytest.raises(ValueError, match="inventory"):
        optimize_price(10, 5, -1, 20, 10, 1.0)
    with pytest.raises(ValueError, match="feasible"):
        optimize_price(10, 9, 10, 10, 10, 1.0, min_margin_floor=0.2)

    result = optimize_price(10, 5, 2, 12, 10, 1.0, min_margin_floor=0.1)

    assert result["optimal_price"] <= 12
    assert result["expected_demand"] <= 2
    assert demand_curve(0, 10, 12, 1) == 0
    with pytest.raises(ValueError, match="finite"):
        demand_curve(1, 10, 12, float("nan"))
