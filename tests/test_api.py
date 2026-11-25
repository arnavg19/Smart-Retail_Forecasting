import asyncio

import pandas as pd
import pytest
from starlette.requests import Request

import api.main as api


def test_health_endpoint_is_registered_and_returns_ok():
    route = next(
        (route for route in api.app.routes if getattr(route, "path", None) == "/health"),
        None,
    )

    assert route is not None
    assert "GET" in route.methods
    assert api.health_check() == {"status": "ok"}


def test_csv_ingestion_reports_measured_counts_and_grid_expansion():
    content = (
        "StockCode,InvoiceDate,Quantity,UnitPrice\n"
        "A,2024-01-01,2,10\n"
        "A,2024-01-03,5,12\n"
        ",invalid,1,1\n"
    ).encode()

    async def receive():
        return {"type": "http.request", "body": content, "more_body": False}

    request = Request(
        {"type": "http", "method": "POST", "path": "/api/ingest", "headers": []},
        receive,
    )
    result = asyncio.run(api.ingest_csv(request, "sales.csv"))

    assert result.input_rows == 3
    assert result.valid_transaction_rows == 2
    assert result.dropped_invalid_rows == 1
    assert result.active_skus == 1
    assert result.grid_rows == 3
    assert result.zero_sale_rows == 1
    assert result.zero_sale_grid_expansion_rows == 1
    assert result.preview[1]["quantity"] == 0.0
    assert result.mode == "live"


def test_artifact_unavailable_forecast_is_explicitly_a_naive_baseline(monkeypatch):
    history = pd.DataFrame(
        {
            "quantity": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "unit_price": [10.0] * 7,
        },
        index=pd.date_range("2024-01-01", periods=7, name="invoice_date"),
    )
    monkeypatch.setattr(api, "_sku_history", lambda _sku: history)
    monkeypatch.setattr(api, "_load_model", lambda: (None, None))

    result = api.get_forecast("A", horizon=2)

    assert result.mode == "live_baseline"
    assert result.model_available is False
    assert result.model == "7-day seasonal naive (no trained artifact)"
    assert [point.prediction for point in result.points[-2:]] == [1.0, 2.0]
    assert [point.baseline for point in result.points[-2:]] == [1.0, 2.0]


def test_portfolio_zero_sale_rate_uses_actual_zero_quantity_rows(monkeypatch):
    data = pd.DataFrame(
        [
            ("A", "2024-01-01", 2.0, 10.0),
            ("A", "2024-01-02", 0.0, 10.0),
            ("A", "2024-01-03", 0.0, 10.0),
            ("B", "2024-01-01", 1.0, 8.0),
            ("B", "2024-01-02", 1.0, 8.0),
            ("B", "2024-01-03", 1.0, 8.0),
        ],
        columns=["item_id", "invoice_date", "quantity", "unit_price"],
    )
    data["invoice_date"] = pd.to_datetime(data["invoice_date"])
    monkeypatch.setattr(api, "_load_daily_data", lambda: (data, "test fixture"))
    monkeypatch.setattr(api, "_load_model", lambda: (None, None))
    monkeypatch.setattr(
        api,
        "_portfolio_metrics",
        lambda: (
            {"wape": 0.1, "rmsle": 0.2, "baseline_wape": 0.1, "baseline_rmsle": 0.2},
            "test baseline",
        ),
    )

    result = api.get_skus()
    rates = {sku.sku_id: sku.zero_sale_rate for sku in result.skus}

    assert rates["A"] == pytest.approx(2 / 3)
    assert rates["B"] == 0.0


def test_sku_catalog_returns_every_available_sku(monkeypatch):
    data = pd.DataFrame(
        [
            ("A", "2024-01-01", 2.0, 10.0),
            ("B", "2024-01-01", 3.0, 20.0),
            ("C", "2024-01-01", 4.0, 30.0),
        ],
        columns=["item_id", "invoice_date", "quantity", "unit_price"],
    )
    data["invoice_date"] = pd.to_datetime(data["invoice_date"])
    monkeypatch.setattr(api, "_load_daily_data", lambda: (data, "test fixture"))
    monkeypatch.setattr(api, "_load_model", lambda: (None, None))
    monkeypatch.setattr(
        api,
        "_portfolio_metrics",
        lambda: (
            {"wape": 0.1, "rmsle": 0.2, "baseline_wape": 0.1, "baseline_rmsle": 0.2},
            "test baseline",
        ),
    )

    result = api.get_skus()

    assert {sku.sku_id for sku in result.skus} == {"A", "B", "C"}


def test_screener_payload_preserves_500_action_limit_and_revenue_order():
    skus = [
        api.SkuRecord(
            sku_id=f"SKU{index:03d}",
            total_units=float(index),
            avg_daily_demand=1.0,
            unit_price=10.0,
            zero_sale_rate=0.0,
            revenue=float(index),
        )
        for index in range(510)
    ]
    catalog = api.SkuResponse(
        portfolio=api.PortfolioMetrics(
            total_skus=510,
            total_units=1.0,
            avg_daily_demand=1.0,
            total_revenue=1.0,
            zero_sale_rate=0.0,
            data_source="test fixture",
            model_available=True,
            mode="live_model",
        ),
        metrics={},
        metrics_source="test fixture",
        skus=skus,
        mode="live_model",
    )

    result = api._build_screener_payload(catalog)

    assert result.total_skus == 510
    assert result.limit == 500
    assert len(result.skus) == 500
    assert result.skus[0].sku_id == "SKU509"
    assert result.skus[-1].sku_id == "SKU010"


def test_price_api_uses_explicit_inventory_and_historical_ceiling(monkeypatch):
    history = pd.DataFrame(
        {
            "quantity": [1.0, 2.0, 1.0, 3.0],
            "unit_price": [10.0, 12.0, 11.0, 13.0],
        },
        index=pd.date_range("2024-01-01", periods=4, name="invoice_date"),
    )
    monkeypatch.setattr(api, "_sku_history", lambda _sku: history)
    request = api.PriceRequest(
        sku_id="A",
        unit_cost=5,
        competitor_index=1,
        min_margin_floor=0.1,
        max_price_delta_pct=0.25,
        inventory=1,
        elasticity=1.2,
    )

    result = api.optimize_sku_price(request)

    assert result.mode == "what_if_simulation"
    assert result.elasticity_source == "user what-if assumption"
    assert result.optimal_price <= result.historical_max_price
    assert result.projected_demand <= result.inventory_cap
    feasible_min = max(
        request.unit_cost * (1 + request.min_margin_floor),
        result.reference_price * (1 - request.max_price_delta_pct),
    )
    feasible_max = min(
        result.historical_max_price,
        result.reference_price * (1 + request.max_price_delta_pct),
    )
    assert all(
        point.feasible == (feasible_min <= point.price <= feasible_max)
        for point in result.profit_curve
    )


def test_price_api_normalizes_valid_default_bounds(monkeypatch):
    history = pd.DataFrame(
        {
            "quantity": [1.0, 2.0, 1.0, 3.0],
            "unit_price": [10.0, 12.0, 11.0, 13.0],
        },
        index=pd.date_range("2024-01-01", periods=4, name="invoice_date"),
    )
    monkeypatch.setattr(api, "_sku_history", lambda _sku: history)
    request = api.PriceRequest(
        sku_id="A",
        unit_cost=20.0,
        competitor_index=1.0,
        min_margin_floor=0.18,
        max_price_delta_pct=0.25,
        inventory=10,
        elasticity=1.5,
    )

    result = api.optimize_sku_price(request)

    assert result.unit_cost <= result.historical_max_price
    assert result.optimal_price <= result.historical_max_price
    assert result.projected_demand <= result.inventory_cap


def test_price_api_returns_friendly_422_for_explicit_impossible_bounds(monkeypatch):
    history = pd.DataFrame(
        {
            "quantity": [1.0, 2.0, 1.0, 3.0],
            "unit_price": [10.0, 12.0, 11.0, 13.0],
        },
        index=pd.date_range("2024-01-01", periods=4, name="invoice_date"),
    )
    monkeypatch.setattr(api, "_sku_history", lambda _sku: history)
    request = api.PriceRequest(
        sku_id="A",
        unit_cost=20.0,
        competitor_index=1.0,
        min_margin_floor=0.18,
        max_price_delta_pct=0.25,
        inventory=10,
        elasticity=1.5,
        strict_bounds=True,
    )

    with pytest.raises(api.HTTPException) as exc:
        api.optimize_sku_price(request)

    assert exc.value.status_code == 422
    assert "No feasible price in range" in str(exc.value.detail)


def test_price_api_returns_sku_specific_reference_and_results(monkeypatch):
    histories = {
        "A": pd.DataFrame(
            {"quantity": [2.0, 3.0, 4.0], "unit_price": [10.0, 11.0, 12.0]},
            index=pd.date_range("2024-01-01", periods=3, name="invoice_date"),
        ),
        "B": pd.DataFrame(
            {"quantity": [5.0, 6.0, 7.0], "unit_price": [20.0, 21.0, 22.0]},
            index=pd.date_range("2024-01-01", periods=3, name="invoice_date"),
        ),
    }
    monkeypatch.setattr(api, "_sku_history", lambda sku: histories[sku])

    results = {}
    for sku_id in ("A", "B"):
        request = api.PriceRequest(
            sku_id=sku_id,
            unit_cost=histories[sku_id]["unit_price"].median() * 0.58,
            competitor_index=1.0,
            min_margin_floor=0.18,
            max_price_delta_pct=0.25,
            inventory=100,
            elasticity=1.5,
        )
        results[sku_id] = api.optimize_sku_price(request)

    assert results["A"].reference_price == pytest.approx(11.0)
    assert results["B"].reference_price == pytest.approx(21.0)
    assert results["A"].optimal_price != results["B"].optimal_price


def test_diagnostics_includes_four_chronological_benchmarks(monkeypatch):
    monkeypatch.setattr(api, "_load_model", lambda: (None, None))
    monkeypatch.setattr(
        api,
        "_portfolio_metrics",
        lambda: (
            {
                "wape": None,
                "rmsle": None,
                "baseline_wape": 0.25,
                "baseline_rmsle": 0.30,
                "ewma_wape": 0.20,
                "ewma_rmsle": 0.27,
                "ridge_wape": 0.18,
                "ridge_rmsle": 0.24,
            },
            "Chronological validation holdout",
        ),
    )

    result = api.get_diagnostics()

    assert [row.model for row in result.benchmarks] == [
        "LightGBM Tweedie",
        "7-day seasonal naive",
        "7-day EWMA",
        "Poisson/Ridge regression",
    ]
    assert result.benchmarks[2].wape == pytest.approx(0.20)
    assert result.benchmarks[3].rmsle == pytest.approx(0.24)


def test_portfolio_metrics_computes_ewma_and_ridge_on_holdout(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    rows = []
    for sku_id, base in (("A", 10.0), ("B", 20.0)):
        for day, date in enumerate(dates):
            rows.append(
                {
                    "item_id": sku_id,
                    "invoice_date": date,
                    "quantity": base + day,
                    "unit_price": 1.0,
                }
            )
    data = pd.DataFrame(rows)
    monkeypatch.setattr(api, "_load_daily_data", lambda: (data, "test fixture"))

    metrics, _ = api._portfolio_metrics()

    assert metrics["ewma_wape"] is not None
    assert metrics["ewma_rmsle"] is not None
    assert metrics["ridge_wape"] is not None
    assert metrics["ridge_rmsle"] is not None
