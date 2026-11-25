from __future__ import annotations

import io
import json
import logging
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sklearn.linear_model import Ridge
from src.data_loader import (
    clean_transactions,
    clean_transactions_with_audit,
    expand_daily_sales_grid,
    load_raw_transactions,
)
from src.predict import _prepare_model_features
from src.pricing_engine import demand_curve, optimize_price
from src.train import rmsle as compute_rmsle, wape as compute_wape


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "lightgbm_tweedie_model.joblib"
METADATA_PATH = MODELS_DIR / "lightgbm_tweedie_metadata.json"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DAILY_DATA_PATH = PROCESSED_DIR / "daily_demand_clean.csv"
FEATURE_DATA_PATH = PROCESSED_DIR / "daily_demand_features.csv"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.json"

_feature_data: pd.DataFrame | None = None
_history_cache: dict[str, pd.DataFrame] = {}
_forecast_cache: OrderedDict[tuple[str, int], ForecastResponse] = OrderedDict()
_price_response_cache: OrderedDict[tuple[object, ...], PriceResponse] = OrderedDict()
_feature_importance_cache: list[FeatureImportance] | None = None
_sku_catalog_cache: SkuResponse | None = None
_sku_catalog_json: bytes | None = None
_screener_cache: ScreenerResponse | None = None
_screener_json: bytes | None = None
_diagnostics_cache: DiagnosticsResponse | None = None
_startup_ready = False


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _feature_data, _feature_importance_cache, _sku_catalog_cache
    global _sku_catalog_json, _screener_cache, _screener_json, _diagnostics_cache
    global _startup_ready

    started = time.perf_counter()
    _load_daily_data.cache_clear()
    _load_model.cache_clear()
    _portfolio_metrics.cache_clear()
    _history_cache.clear()
    _forecast_cache.clear()
    _price_response_cache.clear()
    _sku_catalog_cache = None
    _sku_catalog_json = None
    _screener_cache = None
    _screener_json = None
    _diagnostics_cache = None
    _feature_importance_cache = None
    _feature_data = None
    _startup_ready = False

    try:
        if not DAILY_DATA_PATH.is_file():
            raise FileNotFoundError(f"Processed retail data not found: {DAILY_DATA_PATH}")
        if not FEATURE_DATA_PATH.is_file():
            raise FileNotFoundError(f"Processed feature data not found: {FEATURE_DATA_PATH}")

        daily_data, _ = _load_daily_data()
        _feature_data = pd.read_csv(
            FEATURE_DATA_PATH,
            dtype={"item_id": "string"},
            low_memory=False,
        )
        model, metadata = _load_model()
        if model is None or metadata is None:
            raise RuntimeError(
                f"The trained LightGBM model and metadata are required: {MODEL_PATH}"
            )

        _warm_history_cache(daily_data)
        _feature_importance_cache = _load_feature_importance(model)
        _portfolio_metrics()
        _sku_catalog_cache = _build_sku_catalog()
        _sku_catalog_json = _serialize_model(_sku_catalog_cache)
        _screener_cache = _build_screener_payload(_sku_catalog_cache)
        _screener_json = _serialize_model(_screener_cache)
        _diagnostics_cache = get_diagnostics()

        application.state.daily_data = daily_data
        application.state.feature_data = _feature_data
        application.state.model = model
        application.state.metadata = metadata
        application.state.sku_catalog = _sku_catalog_cache
        application.state.screener = _screener_cache
        LOGGER.info(
            "Smart Retail API warmed: %d SKUs, %d feature rows, startup %.2fs",
            _sku_catalog_cache.portfolio.total_skus,
            len(_feature_data),
            time.perf_counter() - started,
        )
        _startup_ready = True
    except Exception:
        LOGGER.exception("Smart Retail API startup pre-warming failed")
        raise

    try:
        yield
    finally:
        _startup_ready = False


app = FastAPI(title="Smart-Retail Intelligence API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Report readiness after lifespan data and model pre-warming completes."""
    return {"status": "ok"}


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
    mode: str


class SkuResponse(BaseModel):
    portfolio: PortfolioMetrics
    metrics: dict[str, float | None]
    metrics_source: str
    skus: list[SkuRecord]
    mode: str


class ScreenerResponse(BaseModel):
    total_skus: int
    limit: int
    skus: list[SkuRecord]


class ForecastPoint(BaseModel):
    date: str
    actual: float | None = None
    prediction: float | None = None
    p10: float | None = None
    p90: float | None = None
    baseline: float | None = None
    zero_sale: bool = False


class ForecastResponse(BaseModel):
    sku_id: str
    horizon: int
    model: str
    model_available: bool
    points: list[ForecastPoint]
    mode: str


class PriceRequest(BaseModel):
    sku_id: str = Field(min_length=1)
    unit_cost: float = Field(gt=0)
    competitor_index: float = Field(gt=0, le=5)
    min_margin_floor: float = Field(ge=0, lt=1)
    max_price_delta_pct: float = Field(gt=0, le=1)
    inventory: float = Field(ge=0)
    elasticity: float | None = Field(default=None, ge=0, le=10)
    strict_bounds: bool = False


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
    inventory_cap: float
    unit_cost: float
    min_margin_floor: float
    historical_max_price: float
    max_price_delta_pct: float
    elasticity: float
    elasticity_source: str
    mode: str


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
    mode: str
    tweedie_variance_power: float | None = None


class IngestionResponse(BaseModel):
    input_rows: int
    valid_transaction_rows: int
    dropped_invalid_rows: int
    active_skus: int
    calendar_periods: int
    grid_rows: int
    zero_sale_rows: int
    zero_sale_grid_expansion_rows: int
    preview: list[dict[str, object]]
    mode: str


SCREENER_LIMIT = 500


def _serialize_model(value: BaseModel) -> bytes:
    return json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _warm_history_cache(data: pd.DataFrame) -> None:
    for sku_id, group in data.groupby("item_id", sort=False, observed=True):
        history = group[["invoice_date", "quantity", "unit_price"]].copy()
        history = history.set_index("invoice_date").sort_index()
        calendar = pd.date_range(history.index.min(), history.index.max(), freq="D")
        if not history.index.equals(calendar):
            history = history.reindex(calendar)
        history.index.name = "invoice_date"
        history["quantity"] = history["quantity"].fillna(0.0)
        history["unit_price"] = history["unit_price"].ffill().fillna(0.0)
        _history_cache[str(sku_id)] = history


def _load_feature_importance(model: object) -> list[FeatureImportance]:
    if FEATURE_IMPORTANCE_PATH.is_file():
        with FEATURE_IMPORTANCE_PATH.open(encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, list):
            raise ValueError(f"Invalid feature-importance JSON: {FEATURE_IMPORTANCE_PATH}")
        try:
            return [FeatureImportance.model_validate(row) for row in payload]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid feature-importance JSON: {FEATURE_IMPORTANCE_PATH}"
            ) from error

    booster = getattr(model, "booster_", None)
    if booster is None:
        raise TypeError("Trained model does not expose a fitted LightGBM booster.")
    names = booster.feature_name()
    gains = booster.feature_importance(importance_type="gain")
    if len(names) != len(gains) or not names:
        raise ValueError("Model feature names and gain importances are inconsistent.")
    ranked = (
        pd.DataFrame({"feature": names, "gain": gains})
        .sort_values("gain", ascending=False, kind="mergesort")
        .head(20)
    )
    result = [
        FeatureImportance(feature=str(row.feature), gain=float(row.gain))
        for row in ranked.itertuples(index=False)
    ]
    FEATURE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_IMPORTANCE_PATH.write_text(
        json.dumps([row.model_dump(mode="json") for row in result], indent=2),
        encoding="utf-8",
    )
    return result


@lru_cache(maxsize=1)
def _load_daily_data() -> tuple[pd.DataFrame, str]:
    """Load processed pipeline data, falling back to aggregated raw transactions."""
    for path in (DAILY_DATA_PATH, FEATURE_DATA_PATH):
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
                return _complete_daily_grid(_aggregate_daily(frame)), path.name

    raw_files = sorted((DATA_DIR / "raw").glob("*.csv"))
    if not raw_files:
        raise HTTPException(status_code=503, detail="No processed data or raw transaction CSV is available.")
    cleaned = clean_transactions(load_raw_transactions(raw_files[0]))
    if cleaned.empty:
        raise HTTPException(status_code=503, detail="The raw transaction dataset has no valid sales.")
    return _complete_daily_grid(_aggregate_daily(cleaned)), "raw transactions"


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


def _complete_daily_grid(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    start = frame["invoice_date"].min().floor("D")
    end = frame["invoice_date"].max().floor("D")
    items = frame["item_id"].drop_duplicates().sort_values()
    calendar = pd.date_range(start, end, freq="D")
    grid = pd.MultiIndex.from_product(
        [items, calendar], names=["item_id", "invoice_date"]
    ).to_frame(index=False)
    completed = grid.merge(frame, on=["item_id", "invoice_date"], how="left")
    completed["quantity"] = completed["quantity"].fillna(0.0)
    completed["unit_price"] = (
        completed.groupby("item_id", sort=False)["unit_price"]
        .transform(lambda prices: prices.ffill().fillna(0.0))
        .fillna(0.0)
    )
    return completed


@app.post("/api/ingest", response_model=IngestionResponse)
async def ingest_csv(
    request: Request,
    filename: str = Query(min_length=1, max_length=255),
) -> IngestionResponse:
    """Validate, clean, and audit an uploaded CSV without persisting it."""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Only CSV uploads are supported.")
    content = await request.body()
    if not content:
        raise HTTPException(status_code=422, detail="The uploaded CSV is empty.")
    try:
        raw = pd.read_csv(io.BytesIO(content), encoding="ISO-8859-1", low_memory=False)
        cleaned, audit = clean_transactions_with_audit(raw)
    except (UnicodeError, pd.errors.ParserError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"CSV validation failed: {error}") from error
    if cleaned.empty:
        raise HTTPException(status_code=422, detail="No valid sales rows remain after cleaning.")
    daily = expand_daily_sales_grid(
        cleaned, min_sales=1, date_start=cleaned["invoice_date"].min(),
        date_end=cleaned["invoice_date"].max(),
    )
    observed_periods = int(cleaned[["item_id", "invoice_date"]].drop_duplicates().shape[0])
    zero_sales = int(daily["quantity"].eq(0).sum())
    return IngestionResponse(
        **audit,
        calendar_periods=int(daily["invoice_date"].nunique()),
        grid_rows=len(daily),
        zero_sale_rows=zero_sales,
        zero_sale_grid_expansion_rows=max(len(daily) - observed_periods, 0),
        preview=daily.head(20).assign(
            invoice_date=lambda values: values["invoice_date"].dt.date.astype(str)
        ).to_dict(orient="records"),
        mode="live",
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
    if not _is_daily_artifact(metadata):
        return None, None
    model = joblib.load(MODEL_PATH)
    if getattr(model, "booster_", None) is None:
        raise RuntimeError(f"Model artifact is not a fitted LightGBM model: {MODEL_PATH}")
    return model, metadata


def _is_daily_artifact(metadata: dict[str, object]) -> bool:
    frequency = metadata.get("frequency", "D")
    try:
        normalized_frequency = pd.tseries.frequencies.to_offset(str(frequency)).freqstr
    except ValueError:
        LOGGER.warning("Model metadata has an invalid frequency: %r", frequency)
        return False
    if normalized_frequency != "D":
        LOGGER.warning(
            "The API serves daily forecast horizons only; artifact frequency %r is unsupported.",
            normalized_frequency,
        )
        return False
    return True


def _sku_history(sku_id: str) -> pd.DataFrame:
    cached = _history_cache.get(sku_id)
    if cached is not None:
        return cached
    data, _ = _load_daily_data()
    history = data.loc[data["item_id"].astype(str) == sku_id].copy()
    if history.empty:
        raise HTTPException(status_code=404, detail=f"Unknown SKU: {sku_id}")
    history = history.set_index("invoice_date").sort_index()
    calendar = pd.date_range(history.index.min(), history.index.max(), freq="D")
    history = history.reindex(calendar)
    history.index.name = "invoice_date"
    history["quantity"] = history["quantity"].fillna(0.0)
    history["unit_price"] = history["unit_price"].ffill().fillna(0.0)
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


@lru_cache(maxsize=1)
def _portfolio_metrics() -> tuple[dict[str, float | None], str]:
    model, metadata = _load_model()
    if model is not None and metadata is not None:
        metrics = metadata.get("metrics", {})
        if _is_daily_artifact(metadata) and isinstance(metrics, dict):
            model_wape = metrics.get("validation_wape")
            model_rmsle = metrics.get("validation_rmsle")
            ewma_wape = metrics.get("validation_ewma_wape")
            ewma_rmsle = metrics.get("validation_ewma_rmsle")
            ridge_wape = metrics.get("validation_ridge_wape")
            ridge_rmsle = metrics.get("validation_ridge_rmsle")
            if ewma_wape is None or ewma_rmsle is None or ridge_wape is None or ridge_rmsle is None:
                data, _ = _load_daily_data()
                dates = np.sort(data["invoice_date"].dropna().unique())
                if len(dates) >= 2:
                    split = min(max(int(np.ceil(len(dates) * 0.8)), 1), len(dates) - 1)
                    validation_start = dates[split]
                    ordered = data.sort_values(["item_id", "invoice_date"], kind="mergesort").copy()
                    training = ordered.loc[ordered["invoice_date"] < validation_start]
                    validation = ordered.loc[ordered["invoice_date"] >= validation_start].copy()
                    sku_means = training.groupby("item_id")["quantity"].mean()
                    fallback_mean = float(training["quantity"].mean()) if not training.empty else 0.0
                    ewma = ordered.groupby("item_id", sort=False)["quantity"].transform(
                        lambda values: values.ewm(span=7, adjust=False, min_periods=1).mean().shift(1)
                    )
                    ewma_predictions = ewma.loc[validation.index].fillna(validation["item_id"].map(sku_means)).fillna(fallback_mean)
                    ewma_wape = ewma_wape if ewma_wape is not None else float(compute_wape(validation["quantity"], ewma_predictions))
                    ewma_rmsle = ewma_rmsle if ewma_rmsle is not None else float(compute_rmsle(validation["quantity"], ewma_predictions))

                    train_items = pd.Categorical(training["item_id"])
                    valid_items = pd.Categorical(validation["item_id"], categories=train_items.categories)
                    train_dates = (training["invoice_date"] - training["invoice_date"].min()).dt.days.to_numpy(dtype=float)
                    valid_dates = (validation["invoice_date"] - training["invoice_date"].min()).dt.days.to_numpy(dtype=float)
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(
                        np.column_stack([train_items.codes, train_dates]),
                        training["quantity"].to_numpy(dtype=float),
                    )
                    ridge_predictions = np.maximum(
                        ridge.predict(np.column_stack([valid_items.codes, valid_dates])),
                        0.0,
                    )
                    ridge_wape = ridge_wape if ridge_wape is not None else float(compute_wape(validation["quantity"], ridge_predictions))
                    ridge_rmsle = ridge_rmsle if ridge_rmsle is not None else float(compute_rmsle(validation["quantity"], ridge_predictions))
            return {
                "wape": float(model_wape) if model_wape is not None else None,
                "rmsle": float(model_rmsle) if model_rmsle is not None else None,
                "baseline_wape": (
                    float(metrics["validation_baseline_wape"])
                    if metrics.get("validation_baseline_wape") is not None
                    else None
                ),
                "baseline_rmsle": (
                    float(metrics["validation_baseline_rmsle"])
                    if metrics.get("validation_baseline_rmsle") is not None
                    else None
                ),
                "ewma_wape": float(ewma_wape) if ewma_wape is not None else None,
                "ewma_rmsle": float(ewma_rmsle) if ewma_rmsle is not None else None,
                "ridge_wape": float(ridge_wape) if ridge_wape is not None else None,
                "ridge_rmsle": float(ridge_rmsle) if ridge_rmsle is not None else None,
            }, "Chronological validation holdout"

    data, _ = _load_daily_data()
    dates = np.sort(data["invoice_date"].dropna().unique())
    if len(dates) < 2:
        return {
            "wape": None,
            "rmsle": None,
            "baseline_wape": None,
            "baseline_rmsle": None,
            "ewma_wape": None,
            "ewma_rmsle": None,
            "ridge_wape": None,
            "ridge_rmsle": None,
        }, "Metrics unavailable: insufficient history"
    split = min(max(int(np.ceil(len(dates) * 0.8)), 1), len(dates) - 1)
    validation_start = dates[split]
    ordered = data.sort_values(["item_id", "invoice_date"], kind="mergesort").copy()
    ordered["seasonal_naive"] = ordered.groupby("item_id", sort=False)["quantity"].shift(7)
    training = ordered.loc[ordered["invoice_date"] < validation_start]
    validation = ordered.loc[ordered["invoice_date"] >= validation_start].copy()
    sku_means = training.groupby("item_id")["quantity"].mean()
    fallback_mean = float(training["quantity"].mean()) if not training.empty else 0.0
    predictions = validation["seasonal_naive"].fillna(validation["item_id"].map(sku_means)).fillna(fallback_mean)
    baseline_wape = float(compute_wape(validation["quantity"], predictions))
    baseline_rmsle = float(compute_rmsle(validation["quantity"], predictions))
    ewma = ordered.groupby("item_id", sort=False)["quantity"].transform(
        lambda values: values.ewm(span=7, adjust=False, min_periods=1).mean().shift(1)
    )
    ewma_predictions = ewma.loc[validation.index].fillna(validation["item_id"].map(sku_means)).fillna(fallback_mean)
    ewma_wape = float(compute_wape(validation["quantity"], ewma_predictions))
    ewma_rmsle = float(compute_rmsle(validation["quantity"], ewma_predictions))

    training = training.sort_index() if hasattr(training, "sort_index") else training
    train_item_codes = pd.Categorical(training["item_id"]).codes
    valid_item_codes = pd.Categorical(
        validation["item_id"],
        categories=pd.Categorical(training["item_id"]).categories,
    ).codes
    train_dates = (training["invoice_date"] - training["invoice_date"].min()).dt.days.to_numpy(dtype=float)
    valid_dates = (validation["invoice_date"] - training["invoice_date"].min()).dt.days.to_numpy(dtype=float)
    ridge_features_train = np.column_stack([train_item_codes, train_dates])
    ridge_features_valid = np.column_stack([valid_item_codes, valid_dates])
    ridge = Ridge(alpha=1.0)
    ridge.fit(ridge_features_train, training["quantity"].to_numpy(dtype=float))
    ridge_predictions = np.maximum(ridge.predict(ridge_features_valid), 0.0)
    ridge_wape = float(compute_wape(validation["quantity"], ridge_predictions))
    ridge_rmsle = float(compute_rmsle(validation["quantity"], ridge_predictions))
    return {
        "wape": baseline_wape,
        "rmsle": baseline_rmsle,
        "baseline_wape": baseline_wape,
        "baseline_rmsle": baseline_rmsle,
        "ewma_wape": ewma_wape,
        "ewma_rmsle": ewma_rmsle,
        "ridge_wape": ridge_wape,
        "ridge_rmsle": ridge_rmsle,
    }, "7-day naive chronological holdout (baseline only)"


@app.get("/api/skus", response_model=SkuResponse)
def get_skus():
    if _sku_catalog_json is not None:
        return Response(content=_sku_catalog_json, media_type="application/json")
    return _build_sku_catalog()


def _build_sku_catalog() -> SkuResponse:
    data, source = _load_daily_data()
    grouped = data.groupby("item_id", sort=False).agg(
        total_units=("quantity", "sum"),
    )
    observed_prices = data.loc[data["unit_price"] > 0].groupby("item_id")["unit_price"]
    grouped["unit_price"] = observed_prices.median().reindex(grouped.index).fillna(0.0)
    portfolio_start = data["invoice_date"].min()
    portfolio_end = data["invoice_date"].max()
    calendar_days = max(int((portfolio_end - portfolio_start).days) + 1, 1)
    zero_counts = data.assign(zero_sale=data["quantity"].eq(0)).groupby("item_id")[
        "zero_sale"
    ].sum()
    grouped["zero_sale_rate"] = zero_counts.reindex(grouped.index, fill_value=0) / calendar_days
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
            mode="live_model" if model is not None else "live_baseline",
        ),
        metrics=metrics,
        metrics_source=metrics_source,
        skus=[
            SkuRecord(sku_id=str(sku_id), **row)
            for sku_id, row in grouped.to_dict(orient="index").items()
        ],
        mode="live_model" if model is not None else "live_baseline",
    )


@app.get("/api/screener", response_model=ScreenerResponse)
def get_screener(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=SCREENER_LIMIT, ge=1, le=SCREENER_LIMIT),
):
    if _screener_cache is None:
        catalog = _sku_catalog_cache or _build_sku_catalog()
        payload = _build_screener_payload(catalog)
    else:
        payload = _screener_cache
    if offset == 0 and limit == SCREENER_LIMIT and _screener_json is not None:
        return Response(content=_screener_json, media_type="application/json")
    return ScreenerResponse(
        total_skus=payload.total_skus,
        limit=limit,
        skus=payload.skus[offset : offset + limit],
    )


def _build_screener_payload(catalog: SkuResponse) -> ScreenerResponse:
    prioritized = sorted(catalog.skus, key=lambda sku: sku.revenue, reverse=True)
    return ScreenerResponse(
        total_skus=catalog.portfolio.total_skus,
        limit=min(SCREENER_LIMIT, len(prioritized)),
        skus=prioritized[:SCREENER_LIMIT],
    )


@app.get("/api/forecast/{sku_id}", response_model=ForecastResponse)
def get_forecast(
    sku_id: str, horizon: int = Query(default=14, ge=1, le=60)
) -> ForecastResponse:
    cache_key = (sku_id, horizon)
    if _startup_ready and cache_key in _forecast_cache:
        _forecast_cache.move_to_end(cache_key)
        return _forecast_cache[cache_key]
    history = _sku_history(sku_id)
    model, metadata = _load_model()
    last_week = history["quantity"].tail(7).to_numpy(dtype=np.float64)
    seasonal_naive = np.asarray(
        [
            float(last_week[(step - 1) % len(last_week)]) if len(last_week) else 0.0
            for step in range(1, horizon + 1)
        ],
        dtype=np.float64,
    )
    if model is not None and metadata is not None:
        prediction = _forecast_with_artifact(sku_id, history, horizon, model, metadata)
        variance_power = metadata.get("tweedie_variance_power")
        if isinstance(variance_power, (int, float)) and np.isfinite(variance_power):
            model_name = f"LightGBM Tweedie (p={variance_power:g})"
        else:
            model_name = "LightGBM Tweedie"
    else:
        prediction = seasonal_naive.copy()
        model_name = "7-day seasonal naive (no trained artifact)"
    future = []
    for index in range(horizon):
        date = history.index[-1] + pd.Timedelta(days=index + 1)
        value = float(prediction[index])
        future.append(
            ForecastPoint(
                date=date.date().isoformat(),
                prediction=value,
                baseline=float(seasonal_naive[index]),
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
    response = ForecastResponse(
        sku_id=sku_id,
        horizon=horizon,
        model=model_name,
        model_available=model is not None,
        points=observed + future,
        mode="live_model" if model is not None else "live_baseline",
    )
    if _startup_ready:
        _forecast_cache[cache_key] = response
        if len(_forecast_cache) > 2048:
            _forecast_cache.popitem(last=False)
    return response


@app.post("/api/optimize-price", response_model=PriceResponse)
def optimize_sku_price(request: PriceRequest) -> PriceResponse:
    cache_key = (
        request.sku_id,
        request.unit_cost,
        request.competitor_index,
        request.min_margin_floor,
        request.max_price_delta_pct,
        request.inventory,
        request.elasticity,
        request.strict_bounds,
    )
    if _startup_ready and cache_key in _price_response_cache:
        _price_response_cache.move_to_end(cache_key)
        return _price_response_cache[cache_key]
    response = _calculate_sku_price(request)
    if _startup_ready:
        _price_response_cache[cache_key] = response
        if len(_price_response_cache) > 4096:
            _price_response_cache.popitem(last=False)
    return response


def _calculate_sku_price(request: PriceRequest) -> PriceResponse:
    history = _sku_history(request.sku_id)
    demand_base = float(history["quantity"].tail(28).mean())
    observed_prices = history.loc[history["unit_price"] > 0, "unit_price"]
    if observed_prices.empty:
        raise HTTPException(
            status_code=422,
            detail="No observed positive prices are available for this SKU.",
        )
    base_price = float(observed_prices.tail(60).median())
    reference_price = base_price * request.competitor_index
    elasticity = request.elasticity
    elasticity_source = "user what-if assumption"
    if elasticity is None:
        elasticity = _estimate_elasticity(history)
        elasticity_source = "estimated from observed price-sales history"
    if elasticity is None:
        raise HTTPException(
            status_code=422,
            detail="Insufficient stable price variation to estimate elasticity; provide an explicit what-if elasticity.",
        )
    historical_max = float(observed_prices.max())
    effective_cost = request.unit_cost
    effective_margin = request.min_margin_floor
    upper_bound = min(historical_max, reference_price * (1.0 + request.max_price_delta_pct))
    if upper_bound <= 0:
        if request.strict_bounds:
            raise HTTPException(
                status_code=422,
                detail="No feasible price in range. Adjust the cost or price-change bounds.",
            )
        upper_bound = max(historical_max, 0.01)

    cost_limited_ceiling = upper_bound / (1.0 + effective_margin)
    if effective_cost > cost_limited_ceiling:
        if request.strict_bounds:
            raise HTTPException(
                status_code=422,
                detail="No feasible price in range. Unit cost and margin exceed the available price ceiling.",
            )
        effective_cost = cost_limited_ceiling

    lower_bound = max(effective_cost * (1.0 + effective_margin), reference_price * (1.0 - request.max_price_delta_pct))
    if lower_bound > upper_bound:
        if request.strict_bounds:
            raise HTTPException(
                status_code=422,
                detail="No feasible price in range. Adjust the margin floor or maximum price change.",
            )
        lower_bound = upper_bound

    effective_margin = max(0.0, lower_bound / effective_cost - 1.0) if effective_cost > 0 else 0.0
    effective_margin = min(effective_margin, 0.999)
    try:
        engine_result = optimize_price(
            predicted_demand=demand_base,
            unit_cost=effective_cost,
            inventory=request.inventory,
            historical_max_price=historical_max,
            reference_price=reference_price,
            elasticity=elasticity,
            min_margin_floor=effective_margin,
            max_price_delta_pct=request.max_price_delta_pct,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    def demand_at_price(price: float) -> float:
        try:
            return min(
                demand_curve(demand_base, reference_price, price, elasticity),
                request.inventory,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    policy_lower = reference_price * (1.0 - request.max_price_delta_pct)
    policy_upper = reference_price * (1.0 + request.max_price_delta_pct)
    grid = np.linspace(policy_lower, policy_upper, 81)
    curve = []
    for price in grid:
        demand = demand_at_price(float(price))
        curve.append(
            ProfitPoint(
                price=float(price),
                demand=float(demand),
                profit=float((price - effective_cost) * demand),
                feasible=lower_bound <= price <= upper_bound,
            )
        )
    engine_price = float(engine_result["optimal_price"])
    curve_mask = np.asarray([point.feasible for point in curve], dtype=bool)
    candidate_prices = np.append(grid[curve_mask], engine_price)
    profits = [
        (price - request.unit_cost) * demand_at_price(float(price))
        for price in candidate_prices
    ]
    optimal_price = float(candidate_prices[int(np.argmax(profits))])
    projected_demand = demand_at_price(optimal_price)
    optimal_profit = (optimal_price - effective_cost) * projected_demand
    baseline_demand = demand_at_price(reference_price)
    baseline_profit = (reference_price - effective_cost) * baseline_demand
    return PriceResponse(
        sku_id=request.sku_id,
        optimal_price=optimal_price,
        projected_demand=float(projected_demand),
        net_profit_uplift=float(optimal_profit - baseline_profit),
        reference_price=float(reference_price),
        profit_curve=curve,
        inventory_cap=request.inventory,
        unit_cost=effective_cost,
        min_margin_floor=effective_margin,
        historical_max_price=historical_max,
        max_price_delta_pct=request.max_price_delta_pct,
        elasticity=elasticity,
        elasticity_source=elasticity_source,
        mode="what_if_simulation",
    )


def _estimate_elasticity(history: pd.DataFrame) -> float | None:
    valid = history.loc[(history["quantity"] > 0) & (history["unit_price"] > 0)]
    if len(valid) < 3 or valid["unit_price"].nunique() < 2:
        return None
    elasticity = np.polyfit(np.log(valid["unit_price"]), np.log(valid["quantity"]), 1)[0]
    if not np.isfinite(elasticity) or abs(elasticity) > 10:
        return None
    return float(abs(elasticity))


@app.get("/api/diagnostics", response_model=DiagnosticsResponse)
def get_diagnostics() -> DiagnosticsResponse:
    if _diagnostics_cache is not None:
        return _diagnostics_cache
    model, metadata = _load_model()
    metrics, _ = _portfolio_metrics()
    features: list[FeatureImportance] = []
    if model is not None:
        features = _feature_importance_cache or _load_feature_importance(model)
    variance_power = metadata.get("tweedie_variance_power") if metadata else None
    objective = (
        f"Tweedie p={variance_power:g}"
        if isinstance(variance_power, (int, float)) and np.isfinite(variance_power)
        else "Tweedie"
    )
    benchmark_names: list[tuple[str, str]] = [
        ("LightGBM Tweedie", objective),
        ("7-day seasonal naive", "Last observed week repeated"),
        ("7-day EWMA", "Exponentially weighted 7-day demand"),
        ("Poisson/Ridge regression", "Ridge regression with SKU + date signals"),
    ]
    benchmarks = [
        Benchmark(
            model=name,
            objective=objective,
            wape=(
                metrics["wape"] if name == "LightGBM Tweedie" and model is not None
                else metrics["baseline_wape"] if name == "7-day seasonal naive"
                else metrics.get("ewma_wape") if name == "7-day EWMA"
                else metrics.get("ridge_wape") if name == "Poisson/Ridge regression"
                else None
            ),
            rmsle=(
                metrics["rmsle"] if name == "LightGBM Tweedie" and model is not None
                else metrics["baseline_rmsle"] if name == "7-day seasonal naive"
                else metrics.get("ewma_rmsle") if name == "7-day EWMA"
                else metrics.get("ridge_rmsle") if name == "Poisson/Ridge regression"
                else None
            ),
            status=(
                "validation holdout model"
                if name == "LightGBM Tweedie" and model is not None
                else "same validation holdout baseline"
                if name == "7-day seasonal naive"
                else "same validation holdout standalone baseline"
                if name in {"7-day EWMA", "Poisson/Ridge regression"}
                else "trained artifact unavailable"
            ),
        )
        for name, objective in benchmark_names
    ]
    return DiagnosticsResponse(
        model_available=model is not None,
        feature_importance=features,
        benchmarks=benchmarks,
        formula="Q(P) = Q_ref × (P_ref / P)^|ε| ; Π(P) = (P − COGS) × min(Q(P), inventory)",
        mode="live_model" if model is not None else "live_baseline",
        tweedie_variance_power=(
            float(variance_power)
            if isinstance(variance_power, (int, float)) and np.isfinite(variance_power)
            else None
        ),
    )

