# Smart Retail Precision Demand Forecasting & Dynamic Pricing Engine

This repository implements an end-to-end AI/ML system for smart retail demand forecasting and dynamic pricing. It is structured around four technical subsystems and a production delivery layer:

- Data ingestion and time-series grid expansion
- Demand forecasting with machine learning
- Price elasticity and dynamic pricing optimization
- Model explainability and diagnostics

## Project structure

```text
smart-retail-pricing/
├── data/
│   ├── raw/                       # UCI Online Retail II transactional CSV
│   └── processed/                 # Daily zero-filled aggregated dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Data ingestion, cleaning, and zero-sales grid setup
│   ├── features.py                # Lags, rolling statistics, and temporal features
│   ├── forecasting.py             # LightGBM/XGBoost training with Tweedie loss
│   └── pricing_engine.py          # Log-Log price elasticity & scipy optimization
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda_analysis.ipynb
│   ├── 03_demand_forecasting.ipynb
│   └── 04_dynamic_pricing.ipynb
├── reports/
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Core capabilities

### 1) Data ingestion and expansion
The data loader removes invalid transaction rows, standardizes items and dates, and creates a complete daily SKU grid so zero-sale days are explicitly represented.

### 2) Forecasting engine
The forecasting module uses time-aware validation and includes Tweedie-capable gradient boosting logic for retail demand prediction.

### 3) Dynamic pricing
The pricing engine estimates demand elasticity and solves the profit-maximization problem using bounded optimization while applying business guardrails.

### 4) Diagnostics
The system is designed to support SHAP, elasticity analysis, and revenue diagnostics for operational decision-making.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.train
python -m src.evaluate_features
python -m src.predict
```

The training command reads `data/processed/daily_demand_features.csv`, uses a
chronological date-based validation holdout, and saves the LightGBM model and
training metadata under `models/`. Use `--data` and `--models-dir` to override
the default paths.

The evaluation command saves a gain-based feature-importance chart to
`models/feature_importance.png`. Prediction reads
`data/processed/test_features.parquet` when available. If it is absent, it
uses the model's chronological validation period from
`data/processed/daily_demand_features.csv` and warns that these scores are
validation results, not an independent final test. Predictions are written to
`models/test_predictions.csv`.

## VECTRA web terminal

The decoupled FastAPI bridge (`api/`) exposes portfolio, forecast, pricing, and
diagnostic endpoints. It prefers the trained artifacts in `models/` and
processed daily data under `data/processed/`; if no processed table exists, it
aggregates the raw retail transactions. Without a trained model artifact,
forecasts use a labeled Poisson live-data baseline and model-specific metrics
and feature gains remain unavailable. The Vite client (`frontend/`) switches
to deterministic demonstration data when the API is offline.

Run the API from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
Set-Location frontend
npm.cmd install
npm.cmd run dev
```

Open `http://localhost:5173`. The API listens at `http://localhost:8000`, with
interactive documentation at `http://localhost:8000/docs`.
The frontend requires Node.js LTS. After installing Node on Windows, open a
new terminal so it receives the updated `PATH`; use `npm.cmd` in PowerShell
when the execution policy blocks the `npm.ps1` shim.

## Key implementation notes

- Zero-sale expansion is essential because missing values are not the same as true no-demand days.
- Tweedie loss is appropriate for highly sparse retail demand distributions.
- Price optimization includes a minimum margin floor and a historical price ceiling to maintain business control.
- Expanding-window time-series validation prevents leakage between training and testing periods.

## Module summary

- `src/data_loader.py`: cleaning, validation, and daily zero-fill expansion
- `src/features.py`: temporal, lag, and rolling features
- `src/forecasting.py`: WAPE and time-series evaluation utilities
- `src/train.py`: leakage-free LightGBM Tweedie training and artifact persistence
- `src/pricing_engine.py`: elasticity-driven pricing optimization
