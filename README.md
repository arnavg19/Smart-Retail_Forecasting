# Smart Retail Demand Forecasting and Dynamic Pricing

An end-to-end machine learning system that transforms retail transaction data
into daily SKU-level demand forecasts and data-driven price recommendations.
The project combines zero-inflated sales modeling, chronological validation,
profit optimization, and model interpretability.

## Overview

Retail demand is highly variable: many items have zero sales on a given day,
while a smaller number of transactions can represent large quantities or
revenue. This project addresses that behavior with a Tweedie LightGBM model
and evaluates it with retail-relevant error metrics.

The system includes:

- Daily item-level data preparation and zero-demand expansion
- Leakage-safe time-series feature engineering
- LightGBM Tweedie demand forecasting
- Chronological holdout validation
- Elasticity-based bounded price optimization
- Gain-based feature importance reporting
- Automated unit tests and reproducible command-line workflows

## Core Components

### 1. Data Ingestion and Preparation

`src/data_loader.py` loads raw retail CSV data, normalizes common column
names, removes invalid or return transactions, and aggregates sales by SKU
and day.

Missing calendar observations are converted into explicit zero-demand rows.
This prevents a missing transaction record from being interpreted as missing
information in the forecasting model.

### 2. Feature Engineering

`src/features.py` builds leakage-safe inputs for each SKU, including:

- Calendar and cyclical date features
- Historical lag features
- Exponential and rolling demand statistics
- Price and revenue information

All rolling features are computed from historical observations only.

### 3. Demand Forecasting

`src/train.py` trains a LightGBM regression model with Tweedie loss
(`tweedie_variance_power = 1.5`). Tweedie loss is suitable for distributions
that contain many zero values and a right-skewed tail of positive sales.

Training uses a chronological holdout instead of a random split, preserving
the temporal order of observations and reducing the risk of target leakage.

### 4. Dynamic Pricing

`src/pricing_engine.py` applies a log-log demand model to estimate the effect
of price changes. SciPy bounded optimization identifies a candidate price that
maximizes estimated profit while applying the following guardrails:

- A minimum margin relative to unit cost
- A bounded upper price
- Inventory-constrained expected demand
- Positive price and demand constraints

The elasticities and cost assumptions used by the current implementation
should be validated against the business context before production use.

### 5. Evaluation and Explainability

The project evaluates forecasts with:

- WAPE (Weighted Absolute Percentage Error)
- RMSLE (Root Mean Squared Logarithmic Error)
- Gain-based LightGBM feature importance

`src/evaluate_features.py` generates a feature-importance chart, while the
test suite verifies data transformations, model training, and artifact
creation.

## Project Structure

```text
Smart-Retail_Forecasting/
├── .github/workflows/ci.yml
├── src/
│   ├── data_loader.py       # Data cleaning and daily grid expansion
│   ├── features.py          # Time-series feature engineering
│   ├── forecasting.py       # Forecasting and evaluation utilities
│   ├── train.py             # Model training and serialization
│   ├── predict.py           # Holdout/test-set inference
│   ├── pricing_engine.py    # Elasticity and price optimization
│   └── evaluate_features.py # Feature-importance reporting
├── tests/
│   ├── test_pipeline.py
│   └── test_train.py
├── models/
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or later
- pandas and NumPy
- LightGBM
- SciPy
- scikit-learn
- matplotlib
- pyarrow
- pytest

## Installation

Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data Setup

Place the raw retail transaction CSV in `data/raw/`. The ingestion pipeline
accepts the UCI Online Retail II schema and common aliases for the following
fields:

| Required concept | Accepted column names |
| --- | --- |
| SKU identifier | `StockCode`, `stockcode`, `item_id`, `stock_code` |
| Quantity sold | `Quantity`, `quantity` |
| Unit price | `UnitPrice`, `unitprice`, `unit_price`, `price` |
| Transaction date | `InvoiceDate`, `invoicedate`, `invoice_date`, `date` |

The raw and processed datasets are not committed to source control. Generated
inputs and artifacts are stored locally under `data/` and `models/`.

## Running the Pipeline

Run the complete workflow from the repository root:

```bash
python -m pytest -q
python -m src.data_loader
python -m src.features
python -m src.train
python -m src.evaluate_features
python -m src.predict
```

The default training command reads
`data/processed/daily_demand_features.csv` and writes model artifacts to
`models/`.

Common training overrides:

```bash
python -m src.train \
  --data data/processed/daily_demand_features.csv \
  --models-dir models \
  --validation-fraction 0.2 \
  --n-estimators 3000 \
  --early-stopping-rounds 100
```

Prediction reads `data/processed/test_features.parquet` when available. If
the file is absent, it falls back to the chronological validation period
from the daily feature dataset and labels those results as validation
predictions rather than independent test predictions.

## Generated Artifacts

| Artifact | Purpose |
| --- | --- |
| `data/processed/daily_demand_clean.csv` | Clean, expanded daily SKU sales grid |
| `data/processed/daily_demand_features.csv` | Engineered model features |
| `models/lightgbm_tweedie_model.joblib` | Serialized LightGBM demand model |
| `models/lightgbm_tweedie_metadata.json` | Training configuration and feature metadata |
| `models/feature_importance.png` | Gain-based feature-importance chart |
| `models/test_predictions.csv` | Forecast output for test or validation rows |

Actual filenames may vary with command-line overrides.

## Validation

Run the test suite with:

```bash
python -m pytest -q
```

The CI workflow runs the tests through GitHub Actions. The local test suite
covers data transformation, feature integrity, training behavior, and model
artifact serialization.

## Reproducibility Notes

- Time-series validation uses chronological ordering, not random splitting.
- Lag and rolling features must be calculated only from prior observations.
- Retraining parameters should be reviewed whenever the dataset range or
  SKU population changes.
- Forecast quality should be assessed separately by SKU, category, and
  seasonal period where sufficient data is available.
- Dynamic pricing recommendations are decision support; they require
  real costs, inventory, elasticity estimates, and business constraints.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for
details.
