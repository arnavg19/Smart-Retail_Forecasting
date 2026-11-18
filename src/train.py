from __future__ import annotations

import argparse
import json
import logging
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "daily_demand_features.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class TrainingConfig:
    """Paths and hyperparameters for a chronological Tweedie training run."""

    data_path: Path = DEFAULT_DATA_PATH
    models_dir: Path = DEFAULT_MODELS_DIR
    validation_fraction: float = 0.2
    n_estimators: int = 3000
    early_stopping_rounds: int = 100
    learning_rate: float = 0.05
    max_depth: int = 8
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    tweedie_variance_power: float = 1.5
    random_state: int = 42
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if self.n_estimators < 1 or self.early_stopping_rounds < 1:
            raise ValueError("Estimator and early-stopping counts must be positive.")
        if not 1.0 < self.tweedie_variance_power < 2.0:
            raise ValueError("tweedie_variance_power must be between 1 and 2.")


def _metric_arrays(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert metric inputs to validated, aligned one-dimensional arrays."""
    actual = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if actual.size == 0:
        raise ValueError("Metrics require at least one observation.")
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted values must have the same length.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and predicted values must be finite.")
    return actual, predicted


def rmsle(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> float:
    """Compute RMSLE, clipping negative predictions to zero.

    Actual demand must be non-negative. Negative model predictions are clipped
    because the logarithmic metric is defined only for non-negative values.
    """
    actual, predicted = _metric_arrays(y_true, y_pred)
    if (actual < 0.0).any():
        raise ValueError("RMSLE requires non-negative actual demand.")

    predicted = np.maximum(predicted, 0.0)
    log_errors = np.log1p(actual) - np.log1p(predicted)
    return float(np.sqrt(np.mean(np.square(log_errors))))


def wape(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> float:
    """Compute WAPE, returning NaN when total actual volume is zero."""
    actual, predicted = _metric_arrays(y_true, y_pred)
    denominator = np.sum(np.abs(actual))
    if denominator == 0.0:
        return 0.0 if np.sum(np.abs(actual - predicted)) == 0.0 else float("nan")
    return float(np.sum(np.abs(actual - predicted)) / denominator)


def _lightgbm_rmsle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[str, float, bool]:
    """Expose RMSLE as a LightGBM evaluation metric."""
    return "rmsle", rmsle(y_true, y_pred), False


def _lightgbm_wape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[str, float, bool]:
    """Expose WAPE as a LightGBM evaluation metric."""
    return "wape", wape(y_true, y_pred), False


def load_training_data(data_path: str | Path) -> pd.DataFrame:
    """Load and validate the feature-engineered retail demand table.

    Invalid dates, missing targets, and missing item IDs are dropped. Negative
    or non-finite target quantities are rejected because Tweedie demand must be
    finite and non-negative.
    """
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature data not found: {path}")

    data = pd.read_csv(path, dtype={"item_id": "string"}, low_memory=False)
    required = {"item_id", "invoice_date", "quantity"}
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Feature data is missing required columns: {missing_columns}")

    data["invoice_date"] = pd.to_datetime(data["invoice_date"], errors="coerce")
    data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce")
    valid_target = data["quantity"].notna() & np.isfinite(data["quantity"])
    valid_rows = data["invoice_date"].notna() & data["item_id"].notna() & valid_target
    dropped_rows = int((~valid_rows).sum())
    if dropped_rows:
        LOGGER.warning("Dropping %s rows with missing or invalid IDs, dates, or targets", dropped_rows)
    data = data.loc[valid_rows].copy()

    if data.empty:
        raise ValueError("No valid training rows remain after data validation.")
    if (data["quantity"] < 0.0).any():
        raise ValueError("Target quantity must be non-negative for Tweedie training.")

    data["quantity"] = data["quantity"].astype("float64")
    return data.sort_values("invoice_date", kind="mergesort").reset_index(drop=True)


def chronological_split(
    data: pd.DataFrame,
    validation_fraction: float = 0.2,
    date_column: str = "invoice_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split at a date boundary so validation never precedes training data."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if date_column not in data.columns:
        raise ValueError(f"Data must contain a {date_column!r} column.")

    ordered = data.sort_values(date_column, kind="mergesort")
    unique_dates = ordered[date_column].drop_duplicates().to_numpy()
    if unique_dates.size < 2:
        raise ValueError("At least two distinct dates are required for a time-series split.")

    split_index = int(np.ceil((1.0 - validation_fraction) * unique_dates.size))
    split_index = min(max(split_index, 1), unique_dates.size - 1)
    validation_start = unique_dates[split_index]

    train = ordered.loc[ordered[date_column] < validation_start]
    validation = ordered.loc[ordered[date_column] >= validation_start]
    if train.empty or validation.empty:
        raise ValueError("Chronological split produced an empty train or validation set.")
    return train, validation


def _prepare_feature_frames(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Fill feature gaps and encode SKU IDs using training categories only."""
    x_train = train[feature_columns].copy()
    x_validation = validation[feature_columns].copy()
    categorical_features: list[str] = []

    if "item_id" in feature_columns:
        categories = pd.Index(
            x_train["item_id"].astype("string").dropna().unique(),
            dtype="string",
        )
        if categories.empty:
            raise ValueError("No training item IDs are available for categorical encoding.")
        x_train["item_id"] = pd.Categorical(
            x_train["item_id"].astype("string"), categories=categories
        ).codes.astype(np.int32)
        validation_item_ids = x_validation["item_id"].astype("string")
        validation_item_ids = validation_item_ids.where(
            validation_item_ids.isin(categories), pd.NA
        )
        x_validation["item_id"] = pd.Categorical(
            validation_item_ids, categories=categories
        ).codes.astype(np.int32)
        categorical_features.append("item_id")
        item_categories = categories.astype(str).tolist()
    else:
        item_categories = []

    numeric_features = [column for column in feature_columns if column not in categorical_features]
    for frame in (x_train, x_validation):
        if numeric_features:
            numeric = frame[numeric_features].apply(pd.to_numeric, errors="raise")
            frame[numeric_features] = (
                numeric.replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .astype(np.float32)
            )

    return x_train, x_validation, categorical_features, item_categories


def _atomic_joblib_dump(value: object, destination: Path) -> None:
    """Write a joblib artifact in the destination directory, then replace it."""
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        joblib.dump(value, temporary_path, compress=3)
        temporary_path.replace(destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_json_dump(value: dict[str, object], destination: Path) -> None:
    """Write JSON metadata atomically to avoid leaving a partial file."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(value, temporary_file, indent=2, allow_nan=False)
            temporary_file.write("\n")
        if temporary_path is None:
            raise RuntimeError("Could not create a temporary metadata file.")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def train_model(config: TrainingConfig = TrainingConfig()) -> dict[str, object]:
    """Train and persist a LightGBM Tweedie demand model.

    A date-based holdout is used for early stopping and final validation
    metrics. Numeric missing and infinite feature values are replaced with
    zero; item IDs are categorical and categories are learned from training
    dates only. LightGBM trees do not require a scaler, so preprocessing
    details are stored in the metadata artifact instead.
    """
    data = load_training_data(config.data_path)
    excluded = {"quantity", "invoice_date", "gross_revenue"}
    feature_columns = [
        column for column in data.columns if column.lower() not in excluded
    ]
    if not feature_columns:
        raise ValueError("No usable model features were found in the input data.")

    train, validation = chronological_split(data, config.validation_fraction)
    x_train, x_validation, categorical_features, item_categories = _prepare_feature_frames(
        train, validation, feature_columns
    )
    y_train = train["quantity"].to_numpy(dtype=np.float64)
    y_validation = validation["quantity"].to_numpy(dtype=np.float64)

    model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=config.tweedie_variance_power,
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        num_leaves=config.num_leaves,
        subsample=config.subsample,
        subsample_freq=1,
        colsample_bytree=config.colsample_bytree,
        min_child_samples=50,
        reg_lambda=1.0,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        categorical_feature=categorical_features,
        eval_X=x_validation,
        eval_y=y_validation,
        eval_metric=[_lightgbm_rmsle, _lightgbm_wape],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=config.early_stopping_rounds,
                first_metric_only=True,
                verbose=False,
            ),
            lgb.log_evaluation(period=100),
        ],
    )

    train_predictions = model.predict(x_train, num_threads=config.n_jobs)
    validation_predictions = model.predict(x_validation, num_threads=config.n_jobs)
    metrics = {
        "train_rmsle": rmsle(y_train, train_predictions),
        "train_wape": wape(y_train, train_predictions),
        "validation_rmsle": rmsle(y_validation, validation_predictions),
        "validation_wape": wape(y_validation, validation_predictions),
    }
    LOGGER.info(
        "Training scores: RMSLE=%.6f WAPE=%s",
        metrics["train_rmsle"],
        f"{metrics['train_wape']:.6f}" if np.isfinite(metrics["train_wape"]) else "undefined",
    )
    LOGGER.info(
        "Validation scores: RMSLE=%.6f WAPE=%s",
        metrics["validation_rmsle"],
        f"{metrics['validation_wape']:.6f}"
        if np.isfinite(metrics["validation_wape"])
        else "undefined",
    )

    config.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.models_dir / "lightgbm_tweedie_model.joblib"
    metadata_path = config.models_dir / "lightgbm_tweedie_metadata.json"
    _atomic_joblib_dump(model, model_path)

    metadata: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "lightgbm.LGBMRegressor",
        "objective": "tweedie",
        "tweedie_variance_power": config.tweedie_variance_power,
        "target_column": "quantity",
        "date_column": "invoice_date",
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "item_id_categories": item_categories,
        "numeric_missing_value": 0.0,
        "train_start_date": train["invoice_date"].min().isoformat(),
        "train_end_date": train["invoice_date"].max().isoformat(),
        "validation_start_date": validation["invoice_date"].min().isoformat(),
        "validation_end_date": validation["invoice_date"].max().isoformat(),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "best_iteration": int(model.best_iteration_ or config.n_estimators),
        "metrics": {
            key: value if np.isfinite(value) else None for key, value in metrics.items()
        },
        "library_versions": {
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "parameters": asdict(config)
        | {"data_path": str(config.data_path), "models_dir": str(config.models_dir)},
        "preprocessing": {
            "numeric_non_finite_and_missing": "replace with 0.0",
            "unknown_item_id": "encoded as LightGBM missing category (-1)",
            "scaler": "not used; tree-based model",
        },
    }
    _atomic_json_dump(metadata, metadata_path)
    LOGGER.info("Saved model to %s and metadata to %s", model_path, metadata_path)
    return metadata


def _parse_args(argv: Sequence[str] | None = None) -> TrainingConfig:
    """Parse command-line options into a validated training configuration."""
    parser = argparse.ArgumentParser(description="Train a LightGBM Tweedie retail demand model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Feature CSV path.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=3000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    args = parser.parse_args(argv)
    return TrainingConfig(
        data_path=args.data,
        models_dir=args.models_dir,
        validation_fraction=args.validation_fraction,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run training from the command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train_model(_parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
