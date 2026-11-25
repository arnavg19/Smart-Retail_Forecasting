# SmartRetail Intelligence

An end-to-end retail demand forecasting and constrained pricing workspace.
It combines a LightGBM Tweedie forecasting pipeline, a prewarmed FastAPI
service, and a React + TypeScript dashboard.

## Project Overview and Problem Statement

Retail demand is sparse and highly skewed: many SKU-days have no sales, while
positive demand can vary widely. On the current portfolio snapshot, the system
serves **3,033 active SKUs** and approximately **77.5% zero-sale SKU-days**.
It models that distribution with LightGBM Tweedie (`p = 1.5`) and pairs the
forecast with price recommendations constrained by cost, inventory, and
business policy.

## System Architecture and Stack

- **Forecasting engine:** LightGBM with Tweedie objective and variance power
  `p = 1.5`; chronological validation metrics and feature metadata are
  persisted alongside the model.
- **Backend:** FastAPI (`api/main.py`) uses an application lifespan to load
  processed datasets and model artifacts once at startup. It precomputes the
  SKU catalog and screener payload and keeps SKU histories, forecasts, and
  repeated optimizer results in memory.
- **Frontend:** React 18, TypeScript, and Vite provide the interactive
  Forecast Telemetry, Model Attribution, Price Optimizer, and Portfolio
  Screener workspaces.
- **Pipeline:** pandas and NumPy data preparation and feature engineering;
  LightGBM model training; scikit-learn/SciPy evaluation and constrained
  optimization; pytest regression coverage.

The API expects the processed datasets at
`data/processed/daily_demand_clean.csv` and
`data/processed/daily_demand_features.csv`. The trained model and metadata
are stored under `models/`. Dataset CSVs and generated predictions remain
local; the small serving model, metadata, and feature-importance JSON are
tracked.

## Product Modules

### 1. Forecast Telemetry

Shows the LightGBM Tweedie champion's persisted chronological-validation
metrics (**99.89% WAPE**, displayed as **99.9%**; **1.044 RMSLE**), SKU
history, and 7D/14D/30D forecast horizons. The API schema supports optional
P10–P90 uncertainty bounds, but the current live endpoint returns point
forecasts without those bounds. For a Tweedie model, the variance relationship
is $\mathrm{Var}(Y) = \phi \mu^p$; `p = 1.5` supports zero-inflated demand
with a right-skewed positive component.
LightGBM also captures non-linear interactions among lag, calendar, and price
features.

### 2. Model Attribution

Combines split-gain feature attribution with a four-candidate comparison on
the same chronological validation holdout:

| Candidate | WAPE | RMSLE |
| --- | ---: | ---: |
| LightGBM Tweedie (`p = 1.5`) | 99.89% | 1.044 |
| 7-day EWMA | 119.27% | 1.189 |
| 7-day seasonal naive | 127.39% | 1.173 |
| Poisson/Ridge regression | 141.42% | 1.620 |

These are validation results, not an untouched test-set score. The holdout is
used for early stopping; use a separate later-period dataset for independent
evaluation.

### 3. Price Optimizer

Finds a constrained profit-maximizing price using
`(Price - COGS) * min(Q(Price), inventory)`. The optimizer applies unit-cost
and minimum-margin requirements, competitor-price index and maximum price
change bounds, inventory limits, and the historical price ceiling. Elasticity
can be supplied as a what-if assumption or estimated from observed prices;
recommendations are simulations, not causal uplift claims.

### 4. Portfolio Screener

Scores a **revenue-prioritized batch of up to 500 pricing actions** from the
full **3,033-SKU** catalog. The UI windows results for responsive browsing;
the 500-action limit is preserved.

## ML Pipeline Components

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
maximizes estimated profit while enforcing explicit constraints:

- A minimum margin relative to unit cost
- The observed historical maximum price
- An optional reference-price change interval
- Inventory-constrained expected demand
- Finite, non-negative demand inputs and positive prices

The UI identifies costs, inventory, and elasticity as what-if inputs or
estimates. These outputs are simulations, not causal price-uplift claims.

### 5. Evaluation and Explainability

The project evaluates forecasts with:

- WAPE (Weighted Absolute Percentage Error)
- RMSLE (Root Mean Squared Logarithmic Error)
- Gain-based LightGBM feature importance

`src/evaluate_features.py` generates a feature-importance chart, while the
test suite verifies data transformations, model training, and artifact
creation.

## Repository Structure and Architecture

```text
Smart-Retail_Forecasting/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated test workflow
├── api/
│   └── main.py                    # FastAPI application and API endpoints
├── frontend/
│   ├── src/
│   │   ├── App.tsx                # Dashboard UI
│   │   ├── api.ts                 # Live API client with startup retry
│   │   ├── index.css              # Application styles
│   │   ├── main.tsx               # React application entry point
│   │   ├── types.ts               # API response and input types
│   │   └── vite-env.d.ts          # Vite and asset type declarations
│   ├── .gitignore
│   ├── index.html
│   ├── package.json               # Frontend scripts and dependencies
│   ├── package-lock.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   └── vite.config.ts
├── src/                           # Forecasting and pricing pipeline
│   ├── data_loader.py             # Transaction cleaning and daily aggregation
│   ├── evaluate_features.py       # Model feature-importance reporting
│   ├── features.py                # Time-series feature engineering
│   ├── forecasting.py             # Forecast metrics and evaluation utilities
│   ├── predict.py                 # Forecast inference
│   ├── pricing_engine.py          # Elasticity-based price optimization
│   └── train.py                   # Model training and artifact persistence
├── tests/
│   ├── test_api.py                # API, screener, and optimizer tests
│   ├── test_forecasting.py        # Forecast feature and validation tests
│   ├── test_pipeline.py           # Data and feature pipeline tests
│   └── test_train.py              # Training and artifact tests
├── start-local.cmd                # Windows launcher for the PowerShell script
├── start-local.ps1                # Starts and verifies the local services
├── .gitignore
├── LICENSE
├── pytest.ini
├── README.md
└── requirements.txt               # Python dependencies
```

### Component boundaries

- **Forecasting core (`src/`)** owns data preparation, feature engineering,
  training, inference, evaluation, and price optimization.
- **Serving layer (`api/`)** prewarms datasets and artifacts through FastAPI
  lifespan startup, then serves cached catalog, forecast, diagnostics, and
  pricing data.
- **Client (`frontend/`)** renders the retail dashboard and connects only to
  the live API; connection failures are shown explicitly rather than replaced
  with a synthetic portfolio.
- **Quality checks (`tests/`, `.github/workflows/`)** provide local and
  continuous-integration test coverage.

The normal data path is:

```text
Retail transactions -> data preparation -> features -> training/inference
                    -> FastAPI -> React dashboard
```

### Data and model artifact policy

The raw and processed retail datasets are local and excluded from version
control. The trained serving model, its metadata, and the feature-importance
JSON used by the API are tracked. Generated predictions and plots remain local.

```text
├── data/
│   ├── raw/        # Local input transaction data (ignored)
│   └── processed/  # Local cleaned daily data and features (ignored)
└── models/
    ├── lightgbm_tweedie_model.joblib
    ├── lightgbm_tweedie_metadata.json
    └── feature_importance.json
```

Virtual environments, installed frontend dependencies, and frontend build
output are also local/generated and excluded from version control.

## Requirements

- Python 3.10–3.13 (Python 3.14 is not yet verified with the current dependency stack)
- Node.js LTS and npm (frontend install, development, and production build)
- pandas and NumPy
- LightGBM
- SciPy
- scikit-learn
- matplotlib
- pyarrow
- pytest

## Installation

Create an isolated environment and install the dependencies.

```powershell
# Windows PowerShell, from the repository root
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
# macOS or Linux, from the repository root
python3 -m venv .venv
source .venv/bin/activate
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

The raw and processed datasets are not committed to source control. The
LightGBM serving model and its metadata are tracked under `models/`; generated
predictions and plots are ignored.

## Running the Pipeline

Run the complete workflow from the repository root. On Windows, call the
virtual-environment interpreter directly so PowerShell cannot resolve an
unrelated Python installation (for example, MSYS Python):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.data_loader
.\.venv\Scripts\python.exe -m src.features
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m src.evaluate_features
.\.venv\Scripts\python.exe -m src.predict
```

On macOS or Linux, activate `.venv` and run:

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

## Run the Web Workspace

Before starting the API, ensure
`data/processed/daily_demand_clean.csv` and
`data/processed/daily_demand_features.csv` are present. They are intentionally
not committed because of their size; see [Data Setup](#data-setup) and
[Running the Pipeline](#running-the-pipeline). The model and metadata files
under `models/` are tracked. FastAPI preloads the datasets and model artifacts
at startup and reports healthy only after initialization completes.

The API exposes portfolio, forecast, pricing, diagnostics, screener, and
CSV-validation endpoints. `/api/skus` returns the full catalog, and
`/api/screener` returns a revenue-prioritized batch capped at 500 actions. The
Vite client never substitutes a sample catalog when the API is unavailable; it
shows connection status and retries the live API.

### Start both services (Windows PowerShell)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm.cmd --prefix frontend ci
.\start-local.ps1
```

The script starts or reuses the backend and Vite frontend, then checks their
readiness. `.\start-local.cmd` is also available as a Windows launcher.

### Start manually

Run the backend from the repository root:
```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, start Vite:
```powershell
Set-Location frontend
npm.cmd run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173`. Check backend readiness with
`Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"`; interactive API
documentation is at `http://127.0.0.1:8000/docs`. In PowerShell, use `npm.cmd`
if the execution policy blocks the `npm.ps1` shim.

## Key implementation notes

- Zero-sale expansion is essential because missing values are not the same as true no-demand days.
- Tweedie loss is appropriate for highly sparse retail demand distributions.
- Price optimization includes a minimum margin floor and a historical price ceiling to maintain business control.
- Expanding-window time-series validation splits on unique calendar dates, so
  rows for the same date across SKUs stay in one fold.
- A 7-day lag baseline is evaluated on the same chronological holdout as the
  trained model; validation scores are not described as test scores.

Prediction reads `data/processed/test_features.parquet` when available. If
the file is absent, it falls back to the chronological validation period
from the daily feature dataset and labels those results as validation
predictions rather than independent test predictions.

## Generated Artifacts

| Artifact | Purpose |
| --- | --- |
| `data/processed/daily_demand_clean.csv` | Clean, expanded daily SKU sales grid |
| `data/processed/daily_demand_features.csv` | Engineered model features |
| `models/lightgbm_tweedie_model.joblib` | Tracked serialized LightGBM demand model |
| `models/lightgbm_tweedie_metadata.json` | Tracked training configuration, validation metrics, and feature metadata |
| `models/feature_importance.json` | Tracked LightGBM gain importances used by the API |
| `models/feature_importance.png` | Gain-based feature-importance chart |
| `models/test_predictions.csv` | Optional forecast output for an independent test set |
| `models/validation_holdout_predictions.csv` | Optional validation-period predictions, labeled per row |

Only the serving model, metadata, and feature-importance JSON are tracked.
Datasets, prediction files, and plots are generated or supplied locally and
remain ignored by Git.

## Validation

From the repository root, run the Python regression suite:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The suite currently contains 35 tests. Build the production frontend with:

```powershell
Set-Location frontend
npm.cmd run build
```

The frontend build runs TypeScript project checks before Vite bundles the app.
The CI workflow runs the Python tests through GitHub Actions.

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
