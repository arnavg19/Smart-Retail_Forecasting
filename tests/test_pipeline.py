import pandas as pd

from src.data_loader import clean_transactions, expand_daily_sales_grid
from src.features import add_feature_columns
from src.pricing_engine import optimize_price


def test_clean_transactions_removes_invalid_rows():
    df = pd.DataFrame(
        {
            "StockCode": ["A1", "A1", "A2", "A3", "A4"],
            "Quantity": [10, -2, 0, 5, 8],
            "UnitPrice": [12.0, 3.0, 5.0, -1.0, 25.0],
            "InvoiceDate": pd.to_datetime([
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ]),
        }
    )

    cleaned = clean_transactions(df)

    assert list(cleaned["StockCode"]) == ["A1", "A4"]
    assert cleaned["Quantity"].min() > 0
    assert cleaned["UnitPrice"].min() > 0


def test_expand_daily_sales_grid_fills_zero_sales():
    df = pd.DataFrame(
        {
            "StockCode": ["A1", "A1", "A2"],
            "Quantity": [2, 5, 3],
            "UnitPrice": [10.0, 10.0, 11.0],
            "InvoiceDate": pd.to_datetime([
                "2024-01-01",
                "2024-01-03",
                "2024-01-02",
            ]),
        }
    )

    expanded = expand_daily_sales_grid(df, date_start="2024-01-01", date_end="2024-01-03")

    assert set(expanded["StockCode"]) == {"A1", "A2"}
    assert len(expanded) == 6
    assert (expanded["Quantity"] == 0).any()


def test_add_feature_columns_generates_lag_and_rolling_features():
    df = pd.DataFrame(
        {
            "StockCode": ["A1"] * 10,
            "InvoiceDate": pd.date_range("2024-01-01", periods=10, freq="D"),
            "Quantity": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "UnitPrice": [10.0] * 10,
        }
    )

    features = add_feature_columns(df)

    assert "lag_1" in features.columns
    assert "lag_7" in features.columns
    assert "rolling_mean_7" in features.columns
    assert "is_weekend" in features.columns
    assert "price_ratio_30d" in features.columns


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
