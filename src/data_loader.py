from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Final

import pandas as pd


LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


def _clean_column_name(column_name: object) -> str:
    """Normalize DataFrame column names to lowercase snake_case."""
    text = str(column_name).strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()


def load_raw_transactions(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw UCI Online Retail II CSV with the required legacy encoding."""
    path = Path(csv_path)
    LOGGER.info("Loading raw retail transactions from %s", path)

    df = pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)
    if df.empty:
        raise ValueError(f"The source dataset is empty: {path}")

    df = df.copy()
    df.columns = [_clean_column_name(column) for column in df.columns]
    return df


def _normalize_retail_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to the schema required by the ingestion pipeline."""
    normalized = df.copy()

    aliases = {
        "stockcode": "item_id",
        "item_id": "item_id",
        "stock_code": "item_id",
        "quantity": "quantity",
        "unitprice": "unit_price",
        "unit_price": "unit_price",
        "price": "unit_price",
        "invoicedate": "invoice_date",
        "invoice_date": "invoice_date",
        "date": "invoice_date",
    }

    renamed = {column: aliases.get(column, column) for column in normalized.columns}
    normalized = normalized.rename(columns=renamed)

    if "item_id" not in normalized.columns:
        raise ValueError("The dataset must contain an item identifier column such as StockCode or item_id.")
    if "quantity" not in normalized.columns:
        raise ValueError("The dataset must contain a Quantity column.")
    if "unit_price" not in normalized.columns:
        raise ValueError("The dataset must contain a UnitPrice or Price column.")
    if "invoice_date" not in normalized.columns:
        raise ValueError("The dataset must contain an InvoiceDate column.")

    return normalized


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid retail rows and keep only valid daily transaction observations."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["item_id", "invoice_date", "quantity", "unit_price", "gross_revenue"])

    normalized = _normalize_retail_columns(df)
    cleaned = normalized[["item_id", "invoice_date", "quantity", "unit_price"]].copy()

    cleaned["item_id"] = cleaned["item_id"].astype("string").str.strip()
    cleaned["invoice_date"] = pd.to_datetime(cleaned["invoice_date"], errors="coerce")
    cleaned["quantity"] = pd.to_numeric(cleaned["quantity"], errors="coerce")
    cleaned["unit_price"] = pd.to_numeric(cleaned["unit_price"], errors="coerce")

    cleaned = cleaned.dropna(subset=["item_id", "invoice_date", "quantity", "unit_price"]).copy()
    cleaned = cleaned[cleaned["item_id"].astype(str).str.len() > 0]
    cleaned = cleaned[(cleaned["quantity"] > 0) & (cleaned["unit_price"] > 0)].copy()
    cleaned["invoice_date"] = cleaned["invoice_date"].dt.floor("D")
    cleaned["gross_revenue"] = cleaned["quantity"] * cleaned["unit_price"]

    cleaned = cleaned.sort_values(["item_id", "invoice_date"]).reset_index(drop=True)
    LOGGER.info("Cleaned transactions retained %s valid rows", len(cleaned))
    return cleaned


def expand_daily_sales_grid(
    df: pd.DataFrame,
    min_sales: int = 50,
    date_start: str | pd.Timestamp | None = None,
    date_end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a complete daily item-by-date matrix with explicit zero-demand days.

    Retail datasets usually omit non-sale days, which creates temporal missingness for ML models.
    By constructing the full Cartesian product of all active item IDs and the calendar range, the model
    learns that a zero quantity is a real business state rather than a missing observation.
    """
    cleaned = clean_transactions(df)
    if cleaned.empty:
        return pd.DataFrame(columns=["item_id", "invoice_date", "quantity", "unit_price", "gross_revenue"])

    if date_start is None:
        date_start = cleaned["invoice_date"].min().floor("D")
    if date_end is None:
        date_end = cleaned["invoice_date"].max().floor("D")

    start = pd.Timestamp(date_start).floor("D")
    end = pd.Timestamp(date_end).floor("D")
    active_items = (
        cleaned.groupby("item_id", dropna=False)
        .size()
        .loc[lambda counts: counts >= min_sales]
        .index
        .tolist()
    )

    if not active_items:
        LOGGER.warning("No items met the minimum transaction threshold of %s sales. Returning empty grid.", min_sales)
        return pd.DataFrame(columns=["item_id", "invoice_date", "quantity", "unit_price", "gross_revenue"])

    full_calendar = pd.date_range(start=start, end=end, freq="D")
    full_grid = pd.MultiIndex.from_product(
        [sorted(active_items), full_calendar],
        names=["item_id", "invoice_date"],
    ).to_frame(index=False)

    demand_by_day = (
        cleaned.groupby(["item_id", "invoice_date"], as_index=False)["quantity"]
        .sum()
        .rename(columns={"quantity": "quantity"})
    )
    price_by_day = (
        cleaned.groupby(["item_id", "invoice_date"], as_index=False)["unit_price"]
        .mean()
        .rename(columns={"unit_price": "unit_price"})
    )

    daily_grid = full_grid.merge(demand_by_day, on=["item_id", "invoice_date"], how="left")
    daily_grid = daily_grid.merge(price_by_day, on=["item_id", "invoice_date"], how="left")

    daily_grid["quantity"] = daily_grid["quantity"].fillna(0.0).astype(float)
    daily_grid["unit_price"] = daily_grid.groupby("item_id")["unit_price"].transform(
        lambda series: series.ffill().bfill()
    )
    daily_grid["unit_price"] = daily_grid["unit_price"].fillna(0.0)
    daily_grid["gross_revenue"] = daily_grid["quantity"] * daily_grid["unit_price"]

    daily_grid = daily_grid.sort_values(["item_id", "invoice_date"]).reset_index(drop=True)
    LOGGER.info(
        "Expanded daily grid across %s items and %s days with zero-sale rows explicitly retained.",
        len(active_items),
        len(full_calendar),
    )
    return daily_grid


def run_pipeline(
    raw_csv_path: str | Path,
    output_path: str | Path,
    min_sales: int = 50,
) -> pd.DataFrame:
    """Run the full ingestion pipeline and persist the daily demand dataset."""
    raw_path = Path(raw_csv_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_transactions(raw_path)
    processed = expand_daily_sales_grid(raw_df, min_sales=min_sales)
    processed.to_csv(output_file, index=False)
    LOGGER.info("Saved processed daily demand data to %s", output_file)
    return processed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    csv_candidates = sorted(raw_dir.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}. Place the UCI Online Retail II dataset there before running this pipeline.")

    preferred = next(
        (candidate for candidate in csv_candidates if "online" in candidate.name.lower() or "retail" in candidate.name.lower()),
        csv_candidates[0],
    )

    run_pipeline(
        raw_csv_path=preferred,
        output_path=processed_dir / "daily_demand_clean.csv",
        min_sales=50,
    )
