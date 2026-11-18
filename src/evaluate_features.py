from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_tweedie_model.joblib"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "models" / "feature_importance.png"


def evaluate_feature_importance(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    top_n: int = 15,
) -> pd.DataFrame:
    """Print and plot the top model features ranked by LightGBM gain.

    The saved model must be a trusted artifact created by the project's
    training pipeline because joblib files use Python object serialization.

    Args:
        model_path: Path to the serialized LightGBM model.
        output_path: Destination for the high-resolution horizontal bar chart.
        top_n: Maximum number of features to display and plot.

    Returns:
        A DataFrame containing the selected feature names and gain scores.

    Raises:
        FileNotFoundError: If the model artifact does not exist.
        TypeError: If the artifact is not a fitted LightGBM sklearn model.
        ValueError: If ``top_n`` is not positive or feature metadata is invalid.
    """
    model_file = Path(model_path)
    chart_file = Path(output_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Trained model artifact not found: {model_file}")
    if top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    model = joblib.load(model_file)
    booster = getattr(model, "booster_", None)
    if booster is None:
        raise TypeError("Model artifact must be a fitted LightGBM sklearn model.")

    feature_names = booster.feature_name()
    gains = booster.feature_importance(importance_type="gain")
    if len(feature_names) != len(gains) or not feature_names:
        raise ValueError("Model feature names and gain importances are missing or inconsistent.")

    importance = (
        pd.DataFrame({"feature": feature_names, "gain": gains})
        .sort_values("gain", ascending=False, kind="mergesort")
        .head(top_n)
        .reset_index(drop=True)
    )
    print(f"\nTop {len(importance)} features by LightGBM gain")
    print(importance.to_string(index=False, formatters={"gain": "{:,.3f}".format}))

    chart_file.parent.mkdir(parents=True, exist_ok=True)
    chart_data = importance.sort_values("gain", ascending=True)
    figure, axis = plt.subplots(figsize=(11, max(6, 0.42 * len(chart_data))))
    axis.barh(chart_data["feature"], chart_data["gain"], color="#2f6b9a")
    axis.set_title("LightGBM Feature Importance (Gain)")
    axis.set_xlabel("Total gain")
    axis.set_ylabel("Feature")
    axis.grid(axis="x", linestyle="--", alpha=0.35)
    figure.tight_layout()
    figure.savefig(chart_file, dpi=300, bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("Saved feature-importance chart to %s", chart_file)
    return importance


def main(argv: Sequence[str] | None = None) -> int:
    """Run feature-importance evaluation from the command line."""
    parser = argparse.ArgumentParser(description="Plot LightGBM gain feature importances.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    evaluate_feature_importance(args.model, args.output, args.top_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
