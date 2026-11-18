from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from src.train import rmsle, wape


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_tweedie_model.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "lightgbm_tweedie_metadata.json"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_features.parquet"
DEFAULT_FALLBACK_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "daily_demand_features.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "models" / "test_predictions.csv"


def _load_metadata(metadata_path: Path) -> dict[str, object]:
    """Load model preprocessing metadata and validate its required fields."""
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    required_keys = {
        "feature_columns",
        "categorical_features",
        "item_id_categories",
        "target_column",
        "date_column",
    }
    missing_keys = sorted(required_keys.difference(metadata))
    if missing_keys:
        raise ValueError(f"Model metadata is missing required fields: {missing_keys}")
    if not isinstance(metadata["feature_columns"], list):
        raise ValueError("Model metadata field 'feature_columns' must be a list.")
    return metadata


def _prepare_model_features(
    data: pd.DataFrame,
    metadata: dict[str, object],
) -> pd.DataFrame:
    """Apply the same feature ordering, SKU encoding, and numeric filling as training."""
    feature_columns = metadata["feature_columns"]
    categorical_features = metadata["categorical_features"]
    item_categories = metadata["item_id_categories"]
    if not isinstance(feature_columns, list) or not all(
        isinstance(column, str) for column in feature_columns
    ):
        raise ValueError("Model feature column metadata is invalid.")
    if not isinstance(categorical_features, list) or not all(
        isinstance(column, str) for column in categorical_features
    ):
        raise ValueError("Model categorical feature metadata is invalid.")
    if not isinstance(item_categories, list) or not all(
        isinstance(item_id, str) for item_id in item_categories
    ):
        raise ValueError("Model item category metadata is invalid.")

    missing_features = sorted(set(feature_columns).difference(data.columns))
    if missing_features:
        raise ValueError(f"Test data is missing model features: {missing_features}")

    features = data.loc[:, feature_columns].copy()
    for column in categorical_features:
        if column != "item_id":
            raise ValueError(f"Unsupported categorical feature in metadata: {column}")
        if column not in features.columns:
            raise ValueError(f"Categorical feature {column!r} is missing from model features.")
        categories = pd.Index(item_categories, dtype="string")
        item_ids = features[column].astype("string")
        item_ids = item_ids.where(item_ids.isin(categories), pd.NA)
        features[column] = pd.Categorical(
            item_ids, categories=categories
        ).codes.astype(np.int32)

    numeric_features = [
        column for column in feature_columns if column not in categorical_features
    ]
    if numeric_features:
        numeric = features[numeric_features].apply(pd.to_numeric, errors="raise")
        features[numeric_features] = (
            numeric.replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )
    return features


def _load_validation_fallback(
    csv_path: Path,
    metadata: dict[str, object],
) -> pd.DataFrame:
    """Stream the feature CSV and retain only the model's chronological holdout."""
    date_column = metadata["date_column"]
    target_column = metadata["target_column"]
    feature_columns = metadata["feature_columns"]
    validation_start_date = metadata.get("validation_start_date")
    if not isinstance(date_column, str) or not isinstance(target_column, str):
        raise ValueError("Model date and target metadata must be strings.")
    if not isinstance(feature_columns, list) or not all(
        isinstance(column, str) for column in feature_columns
    ):
        raise ValueError("Model feature column metadata is invalid.")
    if not isinstance(validation_start_date, str):
        raise ValueError(
            "Model metadata has no validation_start_date; provide a separate test dataset."
        )

    validation_start = pd.to_datetime(validation_start_date, errors="raise")
    use_columns = list(dict.fromkeys([*feature_columns, date_column, target_column]))
    validation_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=use_columns,
        dtype={"item_id": "string"},
        low_memory=False,
        chunksize=250_000,
    ):
        chunk[date_column] = pd.to_datetime(chunk[date_column], errors="coerce")
        holdout = chunk.loc[chunk[date_column] >= validation_start]
        if not holdout.empty:
            validation_chunks.append(holdout)

    if not validation_chunks:
        raise ValueError(
            "No rows on or after the model validation_start_date were found in "
            f"{csv_path}; supply a separate test dataset or retrain the model."
        )
    return pd.concat(validation_chunks, ignore_index=True)


def predict_test_set(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Predict test demand, report RMSLE/WAPE, and save a row-level CSV.

    The serialized model must be a trusted artifact. Test rows with missing
    IDs, dates, or target values are excluded; negative actual demand is
    rejected because RMSLE is undefined for negative values.

    Args:
        model_path: Serialized model produced by ``src.train``.
        metadata_path: JSON sidecar containing the training feature schema.
        data_path: Test feature dataset in Parquet format.
        output_path: Destination CSV for dates, stock codes, actuals, and
            predictions.

    Returns:
        A pair containing the test metrics and the prediction output frame.

    Raises:
        FileNotFoundError: If the model, metadata, or test data is missing.
        ValueError: If required columns or valid test rows are unavailable.
    """
    model_file = Path(model_path)
    data_file = Path(data_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Trained model artifact not found: {model_file}")

    metadata = _load_metadata(Path(metadata_path))
    target_column = metadata["target_column"]
    date_column = metadata["date_column"]
    if not isinstance(target_column, str) or not isinstance(date_column, str):
        raise ValueError("Target and date column metadata must be strings.")

    using_validation_fallback = False
    if not data_file.is_file():
        if data_file == DEFAULT_DATA_PATH and DEFAULT_FALLBACK_DATA_PATH.is_file():
            LOGGER.warning(
                "Test Parquet file not found at %s; using the model's validation holdout "
                "from %s. For an independent final test score, provide a separate test set.",
                data_file,
                DEFAULT_FALLBACK_DATA_PATH,
            )
            data_file = DEFAULT_FALLBACK_DATA_PATH
            using_validation_fallback = True
        else:
            raise FileNotFoundError(
                f"Test dataset not found: {data_file}. Provide it with --data; when using "
                "the default path, the daily feature CSV is used as a validation fallback."
            )

    if using_validation_fallback:
        data = _load_validation_fallback(data_file, metadata)
    elif data_file.suffix.lower() == ".parquet":
        try:
            data = pd.read_parquet(data_file, engine="pyarrow")
        except ImportError as error:
            raise RuntimeError(
                "Parquet support is unavailable; install the project's pyarrow dependency."
            ) from error
    elif data_file.suffix.lower() == ".csv":
        data = pd.read_csv(data_file, dtype={"item_id": "string"}, low_memory=False)
    else:
        raise ValueError(f"Unsupported test data format: {data_file.suffix}")

    required_columns = {"item_id", date_column, target_column}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Test data is missing required columns: {missing_columns}")

    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data[target_column] = pd.to_numeric(data[target_column], errors="coerce")
    valid_target = data[target_column].notna() & np.isfinite(data[target_column])
    valid_rows = data["item_id"].notna() & data[date_column].notna() & valid_target
    dropped_rows = int((~valid_rows).sum())
    if dropped_rows:
        LOGGER.warning(
            "Dropping %s test rows with missing or invalid IDs, dates, or actual quantities",
            dropped_rows,
        )
    data = data.loc[valid_rows].copy()
    if data.empty:
        raise ValueError("No valid test rows remain after data validation.")
    if (data[target_column] < 0.0).any():
        raise ValueError("Actual test quantities must be non-negative for RMSLE.")
    data = data.sort_values([date_column, "item_id"], kind="mergesort").reset_index(drop=True)

    features = _prepare_model_features(data, metadata)
    model = joblib.load(model_file)
    booster = getattr(model, "booster_", None)
    if booster is None:
        raise TypeError("Model artifact must be a fitted LightGBM sklearn model.")
    if booster.feature_name() != list(features.columns):
        raise ValueError("Prepared test feature order does not match the trained model.")

    predictions = np.asarray(model.predict(features), dtype=np.float64)
    predictions = np.clip(predictions, 0.0, None)
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite test predictions.")

    actual = data[target_column].to_numpy(dtype=np.float64)
    metrics = {
        "rmsle": rmsle(actual, predictions),
        "wape": wape(actual, predictions),
    }
    LOGGER.info("Test RMSLE: %.6f", metrics["rmsle"])
    LOGGER.info(
        "Test WAPE: %s",
        f"{metrics['wape']:.6f}" if np.isfinite(metrics["wape"]) else "undefined",
    )

    prediction_frame = pd.DataFrame(
        {
            "invoice_date": data[date_column],
            "stock_code": data["item_id"].astype("string"),
            "actual_quantity": actual,
            "predicted_quantity": predictions,
        }
    )
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(output_file, index=False)
    LOGGER.info("Saved %s test predictions to %s", len(prediction_frame), output_file)
    return metrics, prediction_frame


def main(argv: Sequence[str] | None = None) -> int:
    """Run test-set inference from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate the trained retail demand model.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    predict_test_set(args.model, args.metadata, args.data, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
