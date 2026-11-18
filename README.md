# Smart Retail Precision Demand Forecasting & Dynamic Pricing Engine

This repository implements an end-to-end AI/ML system for smart retail demand forecasting and dynamic pricing. It is structured around four technical subsystems and a production delivery layer:

- Data ingestion and time-series grid expansion
- Demand forecasting with machine learning
- Price elasticity and dynamic pricing optimization
- Model explainability and diagnostics

## Project structure
```text
Smart-Retail_Forcasting/
├── .github/
│   └── workflows/
│       └── ci.yml             # Automated GitHub Actions test pipeline
├── src/
│   ├── data_loader.py         # Raw data ingestion, anomaly filtering, & schema enforcement
│   ├── features.py            # Leakage-free feature engineering (EWMA lags, cyclic time)
│   ├── train.py               # Chronological time-series split & LightGBM Tweedie training
│   ├── evaluate_features.py   # Gain-based feature importance extraction & plotting
│   ├── predict.py             # Production inference script with defensive fallback streaming
│   ├── pricing_engine.py      # Elasticity-based dynamic pricing optimization logic
│   └── forecasting.py         # Core pipeline orchestration utilities
├── tests/
│   ├── test_pipeline.py       # Unit tests for data transforms and feature integrity
│   └── test_train.py          # Model training and artifact serialization tests
├── models/                    # Serialized model artifacts (.joblib) and feature plots
├── LICENSE                    # MIT Open Source License
├── pytest.ini                 # Pytest root path resolution configuration
├── requirements.txt           # Pinned project dependencies
└── README.md                  # Enterprise portfolio documentation
```
## 🌟 Core Capabilities & Architectural Pillars

<<<<<<< HEAD
#### 1. Zero-Inflated Data Ingestion & Grid Expansion (`src/data_loader.py`)
* **The Problem It Solves:** Retail transaction logs are notoriously sparse, filled with returns, cancellations, and irregular purchasing intervals. 
* **Our Solution:** An automated ingestion pipeline that strips transactional anomalies, preserves foreign-key integrity, and programmatically constructs a **complete daily SKU grid**—ensuring zero-sale days are explicitly modeled rather than silently dropped.

#### 2. Compound Poisson-Gamma Tweedie Forecasting (`src/train.py`)
* **The Problem It Solves:** Standard regression (MSE) treats zero sales as continuous noise, while Poisson models suffer from severe underdispersion in volatile retail environments.
* **Our Solution:** A unified gradient-boosting architecture powered by **LightGBM Tweedie Loss ($p=1.5$)**. This mathematically bridges discrete probability mass at zero with continuous, right-skewed sales volumes in a single, robust training pass under strict chronological splits.

#### 3. Elasticity-Driven Dynamic Pricing Engine (`src/pricing_engine.py`)
* **The Problem It Solves:** Static pricing leads to lost revenue during high-demand surges and excess inventory accumulation during dormant periods.
* **Our Solution:** A business intelligence layer that estimates **log-log price elasticity** per SKU and solves advanced profit-maximization problems via bounded SciPy optimization, backed by hard operational guardrails.

#### 4. Rigorous MLOps Validation & Explainability (`src/evaluate_features.py`)
=======
#### 1. **Zero-Inflated Data Ingestion & Grid Expansion (`src/data_loader.py`)**
* **The Problem It Solves:** Retail transaction logs are notoriously sparse, filled with returns, cancellations, and irregular purchasing intervals. 
* **Our Solution:** An automated ingestion pipeline that strips transactional anomalies, preserves foreign-key integrity, and programmatically constructs a **complete daily SKU grid**—ensuring zero-sale days are explicitly modeled rather than silently dropped.

#### 2. **Compound Poisson-Gamma Tweedie Forecasting (`src/train.py`)**
* **The Problem It Solves:** Standard regression (MSE) treats zero sales as continuous noise, while Poisson models suffer from severe underdispersion in volatile retail environments.
* **Our Solution:** A unified gradient-boosting architecture powered by **LightGBM Tweedie Loss ($p=1.5$)**. This mathematically bridges discrete probability mass at zero with continuous, right-skewed sales volumes in a single, robust training pass under strict chronological splits.

#### 3. **Elasticity-Driven Dynamic Pricing Engine (`src/pricing_engine.py`)**
* **The Problem It Solves:** Static pricing leads to lost revenue during high-demand surges and excess inventory accumulation during dormant periods.
* **Our Solution:** A business intelligence layer that estimates **log-log price elasticity** per SKU and solves advanced profit-maximization problems via bounded SciPy optimization, backed by hard operational guardrails.

#### 4. **Rigorous MLOps Validation & Explainability (`src/evaluate_features.py`)**
>>>>>>> 20e98b7 (docs: upgrade core capabilities section with executive-level framing)
* **The Problem It Solves:** Black-box models fail academic defense and enterprise trust requirements.
* **Our Solution:** Transparent feature attribution via **gain-based importance extraction**, tracked against zero-safe cumulative metrics (**WAPE**) and volume-scaling error bounds (**RMSLE**), all backed by an automated CI test suite.

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
