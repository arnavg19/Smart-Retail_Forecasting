from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.linear_model import PoissonRegressor

from src.data_loader import clean_transactions, load_raw_transactions
from src.evaluate_features import evaluate_feature_importance
from src.predict import _prepare_model_features, predict_test_set
from src.pricing_engine import demand_curve, optimize_price
from src.train import rmsle as compute_rmsle, wape as compute_wape


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "lightgbm_tweedie_model.joblib"
METADATA_PATH = MODELS_DIR / "lightgbm_tweedie_metadata.json"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_PATHS = (
    DATA_DIR / "processed" / "daily_demand_features.csv",
    DATA_DIR / "processed" / "daily_demand_clean.csv",
)

app = FastAPI(title="VECTRA Retail Quant System API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class SkuRecord(BaseModel):
    sku_id: str
    total_units: float
    avg_daily_demand: float
    unit_price: float
    zero_sale_rate: float
    revenue: float


class PortfolioMetrics(BaseModel):
    total_skus: int
    total_units: float
    avg_daily_demand: float
    total_revenue: float
    zero_sale_rate: float
    data_source: str
    model_available: bool


class SkuResponse(BaseModel):
    portfolio: PortfolioMetrics
    metrics: dict[str, float | None]
    metrics_source: str
    skus: list[SkuRecord]


class ForecastPoint(BaseModel):
    date: str
    actual: float | None = None
    prediction: float | None = None
    p10: float | None = None
    p90: float | None = None
    ols: float | None = None
    poisson: float | None = None
    zero_sale: bool = False


class ForecastResponse(BaseModel):
    sku_id: str
    horizon: int
    model: str
    model_available: bool
    points: list[ForecastPoint]


class PriceRequest(BaseModel):
    sku_id: str = Field(min_length=1)
    unit_cost: float = Field(gt=0)
    competitor_index: float = Field(gt=0, le=5)
    min_margin_floor: float = Field(ge=0, lt=1)
    max_price_delta_pct: float = Field(gt=0, le=1)


class ProfitPoint(BaseModel):
    price: float
    demand: float
    profit: float
    feasible: bool


class PriceResponse(BaseModel):
    sku_id: str
    optimal_price: float
    projected_demand: float
    net_profit_uplift: float
    reference_price: float
    profit_curve: list[ProfitPoint]


class FeatureImportance(BaseModel):
    feature: str
    gain: float


class Benchmark(BaseModel):
    model: str
    objective: str
    wape: float | None
    rmsle: float | None
    status: str


class DiagnosticsResponse(BaseModel):
    model_available: bool
    feature_importance: list[FeatureImportance]
    benchmarks: list[Benchmark]
    formula: str


@lru_cache(maxsize=1)
def _load_daily_data() -> tuple[pd.DataFrame, str]:
    """Load processed pipeline data, falling back to aggregated raw transactions."""
    for path in PROCESSED_PATHS:
        if path.is_file():
            frame = pd.read_csv(path, dtype={"item_id": "string"}, low_memory=False)
            required = {"item_id", "invoice_date", "quantity"}
            if required.issubset(frame.columns):
                frame["invoice_date"] = pd.to_datetime(frame["invoice_date"], errors="coerce")
                frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
                frame = frame.dropna(subset=["item_id", "invoice_date", "quantity"])
                if "unit_price" not in frame:
                    frame["unit_price"] = 0.0
                frame["unit_price"] = pd.to_numeric(frame["unit_price"], errors="coerce").fillna(0.0)
                return _aggregate_daily(frame), path.name

    raw_files = sorted((DATA_DIR / "raw").glob("*.csv"))
    if not raw_files:
        raise HTTPException(status_code=503, detail="No processed data or raw transaction CSV is available.")
    cleaned = clean_transactions(load_raw_transactions(raw_files[0]))
    if cleaned.empty:
        raise HTTPException(status_code=503, detail="The raw transaction dataset has no valid sales.")
    return _aggregate_daily(cleaned), "raw transactions"


def _aggregate_daily(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["item_id"] = data["item_id"].astype("string").str.strip()
    data["invoice_date"] = pd.to_datetime(data["invoice_date"], errors="coerce").dt.floor("D")
    data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce")
    data["unit_price"] = pd.to_numeric(data["unit_price"], errors="coerce").fillna(0.0)
    data = data.dropna(subset=["item_id", "invoice_date", "quantity"])
    data = data[data["item_id"].str.len().fillna(0) > 0]
    return (
        data.groupby(["item_id", "invoice_date"], as_index=False, sort=True)
        .agg(quantity=("quantity", "sum"), unit_price=("unit_price", "mean"))
        .sort_values(["item_id", "invoice_date"], kind="mergesort")
        .reset_index(drop=True)
    )


@lru_cache(maxsize=1)
def _load_model() -> tuple[object | None, dict[str, object] | None]:
    if not MODEL_PATH.is_file():
        return None, None
    if not METADATA_PATH.is_file():
        LOGGER.warning("Model file exists without its metadata sidecar: %s", METADATA_PATH)
        return None, None
    with METADATA_PATH.open(encoding="utf-8") as file:
        metadata = json.load(file)
    model = joblib.load(MODEL_PATH)
    if getattr(model, "booster_", None) is None:
        raise RuntimeError(f"Model artifact is not a fitted LightGBM model: {MODEL_PATH}")
    return model, metadata


def _sku_history(sku_id: str) -> pd.DataFrame:
    data, _ = _load_daily_data()
    history = data.loc[data["item_id"].astype(str) == sku_id].copy()
    if history.empty:
        raise HTTPException(status_code=404, detail=f"Unknown SKU: {sku_id}")
    history = history.set_index("invoice_date").sort_index()
    calendar = pd.date_range(history.index.min(), history.index.max(), freq="D")
    history = history.reindex(calendar)
    history.index.name = "invoice_date"
    history["quantity"] = history["quantity"].fillna(0.0)
    history["unit_price"] = history["unit_price"].ffill().bfill().fillna(0.0)
    return history


def _forecast_with_artifact(
    sku_id: str, history: pd.DataFrame, horizon: int, model: object, metadata: dict[str, object]
) -> np.ndarray:
    """Run recursive predictions using the stored model's declared feature schema."""
    recent = history["quantity"].tail(30).to_numpy(dtype=np.float64)
    last_row = _feature_row_for_sku(sku_id, history, metadata)
    predictions: list[float] = []
    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list):
        raise ValueError("Model metadata does not contain a valid feature_columns list.")
    for step in range(1, horizon + 1):
        date = history.index[-1] + pd.Timedelta(days=step)
        row = last_row.copy()
        _update_calendar_features(row, date)
        prior = np.concatenate([recent, np.asarray(predictions, dtype=np.float64)])
        for lag in (1, 7, 14, 30):
            column = f"lag_{lag}_days"
            if column in row:
                row[column] = float(prior[-lag]) if prior.size >= lag else 0.0
        for window in (7, 30):
            values = prior[-window:]
            for suffix, value in (
                ("mean_qty", float(values.mean()) if values.size else 0.0),
                ("std_qty", float(values.std()) if values.size else 0.0),
                ("min_qty", float(values.min()) if values.size else 0.0),
                ("max_qty", float(values.max()) if values.size else 0.0),
            ):
                column = f"rolling_{window}d_{suffix}"
                if column in row:
                    row[column] = value
        if "ewma_7d_qty" in row:
            row["ewma_7d_qty"] = float(np.mean(prior[-7:])) if prior.size else 0.0
        row["item_id"] = sku_id
        features = _prepare_model_features(
            pd.DataFrame([row]), metadata
        )
        prediction = float(np.asarray(model.predict(features)).reshape(-1)[0])
        predictions.append(max(prediction, 0.0))
    return np.asarray(predictions, dtype=np.float64)


def _feature_row_for_sku(
    sku_id: str, history: pd.DataFrame, metadata: dict[str, object]
) -> dict[str, object]:
    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not all(isinstance(col, str) for col in feature_columns):
        raise ValueError("Model metadata contains invalid feature columns.")
    row: dict[str, object] = {column: 0.0 for column in feature_columns}
    row["item_id"] = sku_id
    row["unit_price"] = float(history["unit_price"].iloc[-1])
    _update_calendar_features(row, history.index[-1])
    return row


def _update_calendar_features(row: dict[str, object], date: pd.Timestamp) -> None:
    values: dict[str, float] = {
        "month": float(date.month),
        "day_of_week": float(date.dayofweek),
        "day_of_month": float(date.day),
        "quarter": float(date.quarter),
        "days_since_last_sale": 0.0,
        "month_sin": float(np.sin(2.0 * np.pi * date.month / 12.0)),
        "month_cos": float(np.cos(2.0 * np.pi * date.month / 12.0)),
        "dow_sin": float(np.sin(2.0 * np.pi * date.dayofweek / 7.0)),
        "dow_cos": float(np.cos(2.0 * np.pi * date.dayofweek / 7.0)),
        "is_weekend": float(date.dayofweek >= 5),
        "payday_flag": float(1 <= date.day <= 5),
    }
    row.update({key: value for key, value in values.items() if key in row})


def _baseline_forecasts(history: pd.Series, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    values = history.tail(60).to_numpy(dtype=np.float64)
    x = np.arange(values.size, dtype=np.float64)
    if values.size >= 2:
        slope, intercept = np.polyfit(x, values, 1)
    else:
        slope, intercept = 0.0, float(values.mean()) if values.size else 0.0
    future_x = np.arange(values.size, values.size + horizon, dtype=np.float64)
    ols = np.maximum(intercept + slope * future_x, 0.0)

    poisson = np.full(horizon, max(float(values.mean()) if values.size else 0.0, 0.0))
    if values.size >= 3 and np.any(values > 0):
        design = np.column_stack(
            [x / max(values.size, 1), np.sin(2 * np.pi * x / 7), np.cos(2 * np.pi * x / 7)]
        )
        future_design = np.column_stack(
            [
                future_x / max(values.size, 1),
                np.sin(2 * np.pi * future_x / 7),
                np.cos(2 * np.pi * future_x / 7),
            ]
        )
        estimator = PoissonRegressor(alpha=0.1, max_iter=200)
        estimator.fit(design, values)
        poisson = np.maximum(estimator.predict(future_design), 0.0)
    return ols, poisson


def _portfolio_metrics() -> tuple[dict[str, float | None], str]:
    metadata_file = METADATA_PATH
    if MODEL_PATH.is_file() and metadata_file.is_file():
        with metadata_file.open(encoding="utf-8") as file:
            metadata = json.load(file)
        metrics = metadata.get("metrics", {})
        if isinstance(metrics, dict):
            model_wape = metrics.get("validation_wape")
            model_rmsle = metrics.get("validation_rmsle")
            return {
                "wape": float(model_wape) if model_wape is not None else None,
                "rmsle": float(model_rmsle) if model_rmsle is not None else None,
            }, "LightGBM chronological validation"

    data, _ = _load_daily_data()
    dates = np.sort(data["invoice_date"].dropna().unique())
    if len(dates) < 2:
        return {"wape": None, "rmsle": None}, "Metrics unavailable: insufficient history"
    split = min(max(int(np.ceil(len(dates) * 0.8)), 1), len(dates) - 1)
    validation_start = dates[split]
    ordered = data.sort_values(["item_id", "invoice_date"], kind="mergesort").copy()
    ordered["seasonal_naive"] = ordered.groupby("item_id", sort=False)["quantity"].shift(7)
    training = ordered.loc[ordered["invoice_date"] < validation_start]
    validation = ordered.loc[ordered["invoice_date"] >= validation_start].copy()
    sku_means = training.groupby("item_id")["quantity"].mean()
    fallback_mean = float(training["quantity"].mean()) if not training.empty else 0.0
    predictions = validation["seasonal_naive"].fillna(validation["item_id"].map(sku_means)).fillna(fallback_mean)
    return {
        "wape": float(compute_wape(validation["quantity"], predictions)),
        "rmsle": float(compute_rmsle(validation["quantity"], predictions)),
    }, "7-day seasonal-naive live-data holdout"


@app.get("/api/skus", response_model=SkuResponse)
def get_skus() -> SkuResponse:
    data, source = _load_daily_data()
    grouped = data.groupby("item_id", sort=False).agg(
        total_units=("quantity", "sum"),
        unit_price=("unit_price", "mean"),
    )
    portfolio_start = data["invoice_date"].min()
    portfolio_end = data["invoice_date"].max()
    calendar_days = max(int((portfolio_end - portfolio_start).days) + 1, 1)
    counts = data.groupby("item_id")["quantity"].size()
    grouped["zero_sale_rate"] = 1.0 - counts / calendar_days
    grouped["avg_daily_demand"] = grouped["total_units"] / calendar_days
    grouped["revenue"] = data.assign(
        revenue=data["quantity"] * data["unit_price"]
    ).groupby("item_id")["revenue"].sum()
    grouped = grouped.sort_values("total_units", ascending=False)
    model, _ = _load_model()
    metrics, metrics_source = _portfolio_metrics()
    return SkuResponse(
        portfolio=PortfolioMetrics(
            total_skus=int(len(grouped)),
            total_units=float(grouped["total_units"].sum()),
            avg_daily_demand=float(data["quantity"].sum() / calendar_days),
            total_revenue=float(grouped["revenue"].sum()),
            zero_sale_rate=float(grouped["zero_sale_rate"].mean()) if len(grouped) else 0.0,
            data_source=source,
            model_available=model is not None,
        ),
        metrics=metrics,
        metrics_source=metrics_source,
        skus=[
            SkuRecord(sku_id=str(sku_id), **row)
            for sku_id, row in grouped.head(500).round(4).to_dict(orient="index").items()
        ],
    )


@app.get("/api/forecast/{sku_id}", response_model=ForecastResponse)
def get_forecast(
    sku_id: str, horizon: int = Query(default=14, ge=1, le=60)
) -> ForecastResponse:
    history = _sku_history(sku_id)
    model, metadata = _load_model()
    ols, poisson = _baseline_forecasts(history["quantity"], horizon)
    if model is not None and metadata is not None:
        prediction = _forecast_with_artifact(sku_id, history, horizon, model, metadata)
        model_name = "LightGBM Tweedie (p=1.5)"
    else:
        prediction = poisson.copy()
        model_name = "Poisson live-data fallback (no trained artifact)"
    residual_scale = float(history["quantity"].tail(60).std(ddof=0))
    width = max(residual_scale * 1.2816, 0.0)
    future = []
    for index in range(horizon):
        date = history.index[-1] + pd.Timedelta(days=index + 1)
        value = float(prediction[index])
        future.append(
            ForecastPoint(
                date=date.date().isoformat(),
                prediction=value,
                p10=max(value - width, 0.0),
                p90=value + width,
                ols=float(ols[index]),
                poisson=float(poisson[index]),
            )
        )
    observed = [
        ForecastPoint(
            date=date.date().isoformat(),
            actual=float(row.quantity),
            zero_sale=bool(row.quantity == 0),
        )
        for date, row in history.tail(60).iterrows()
    ]
    return ForecastResponse(
        sku_id=sku_id,
        horizon=horizon,
        model=model_name,
        model_available=model is not None,
        points=observed + future,
    )


@app.post("/api/optimize-price", response_model=PriceResponse)
def optimize_sku_price(request: PriceRequest) -> PriceResponse:
    history = _sku_history(request.sku_id)
    demand_base = max(float(history["quantity"].tail(28).mean()), 1e-6)
    observed_prices = history.loc[history["unit_price"] > 0, "unit_price"]
    base_price = float(observed_prices.tail(60).median()) if not observed_prices.empty else request.unit_cost * 1.5
    reference_price = base_price * request.competitor_index
    elasticity = _estimate_elasticity(history)
    historical_max = max(float(observed_prices.max()) if not observed_prices.empty else reference_price, request.unit_cost * 1.01)
    engine_result = optimize_price(
        predicted_demand=demand_base,
        unit_cost=request.unit_cost,
        inventory=max(demand_base * 2.0, 1.0),
        historical_max_price=historical_max,
        reference_price=reference_price,
        elasticity=elasticity,
    )
    lower_bound = max(request.unit_cost * (1.0 + request.min_margin_floor), reference_price * (1.0 - request.max_price_delta_pct))
    upper_bound = reference_price * (1.0 + request.max_price_delta_pct)
    if lower_bound > upper_bound:
        raise HTTPException(status_code=422, detail="Margin floor exceeds the allowed price-change range.")

    grid = np.linspace(lower_bound, upper_bound, 81)
    curve = []
    for price in grid:
        demand = min(
            demand_curve(demand_base, reference_price, float(price), elasticity),
            max(demand_base * 2.0, 1.0),
        )
        curve.append(
            ProfitPoint(
                price=float(price),
                demand=float(demand),
                profit=float((price - request.unit_cost) * demand),
                feasible=True,
            )
        )
    engine_price = float(engine_result["optimal_price"])
    candidate_prices = np.append(grid, np.clip(engine_price, lower_bound, upper_bound))
    profits = [
        (price - request.unit_cost)
        * min(demand_curve(demand_base, reference_price, float(price), elasticity), max(demand_base * 2.0, 1.0))
        for price in candidate_prices
    ]
    optimal_price = float(candidate_prices[int(np.argmax(profits))])
    projected_demand = min(
        demand_curve(demand_base, reference_price, optimal_price, elasticity),
        max(demand_base * 2.0, 1.0),
    )
    optimal_profit = (optimal_price - request.unit_cost) * projected_demand
    baseline_demand = min(demand_curve(demand_base, reference_price, reference_price, elasticity), max(demand_base * 2.0, 1.0))
    baseline_profit = (reference_price - request.unit_cost) * baseline_demand
    return PriceResponse(
        sku_id=request.sku_id,
        optimal_price=optimal_price,
        projected_demand=float(projected_demand),
        net_profit_uplift=float(optimal_profit - baseline_profit),
        reference_price=float(reference_price),
        profit_curve=curve,
    )


def _estimate_elasticity(history: pd.DataFrame) -> float:
    valid = history.loc[(history["quantity"] > 0) & (history["unit_price"] > 0)]
    if len(valid) < 3 or valid["unit_price"].nunique() < 2:
        return 1.2
    elasticity = np.polyfit(np.log(valid["unit_price"]), np.log(valid["quantity"]), 1)[0]
    return float(np.clip(abs(elasticity), 0.1, 4.0))


@app.get("/api/diagnostics", response_model=DiagnosticsResponse)
def get_diagnostics() -> DiagnosticsResponse:
    model, _ = _load_model()
    metrics, _ = _portfolio_metrics()
    features: list[FeatureImportance] = []
    if model is not None:
        importance = evaluate_feature_importance(
            MODEL_PATH,
            output_path=MODELS_DIR / "feature_importance.png",
            top_n=20,
        )
        features = [
            FeatureImportance(feature=str(row.feature), gain=float(row.gain))
            for row in importance.itertuples(index=False)
        ]
    benchmark_names: list[tuple[str, str]] = [
        ("LightGBM Tweedie", "Tweedie p=1.5"),
        ("Poisson GLM", "Poisson deviance"),
        ("OLS trend", "Squared error"),
        ("Seasonal naive", "7-day seasonal"),
    ]
    benchmarks = [
        Benchmark(
            model=name,
            objective=objective,
            wape=metrics["wape"] if name == "LightGBM Tweedie" and model is not None else None,
            rmsle=metrics["rmsle"] if name == "LightGBM Tweedie" and model is not None else None,
            status="trained artifact" if name == "LightGBM Tweedie" and model is not None else (
                "metrics unavailable" if name == "LightGBM Tweedie" else "benchmark available per SKU"
            ),
        )
        for name, objective in benchmark_names
    ]
    if model is None and (MODEL_PATH.is_file() or METADATA_PATH.is_file()):
        LOGGER.warning("Incomplete or absent trained model artifacts; attempting pipeline evaluation.")
        if MODEL_PATH.is_file():
            try:
                predict_test_set()
            except FileNotFoundError as error:
                LOGGER.warning("Pipeline prediction cannot run without complete artifacts: %s", error)
    return DiagnosticsResponse(
        model_available=model is not None,
        feature_importance=features,
        benchmarks=benchmarks,
        formula="Q(P) = Q_ref × (P_ref / P)^|ε| ; Π(P) = (P − COGS) × min(Q(P), inventory)",
    )

