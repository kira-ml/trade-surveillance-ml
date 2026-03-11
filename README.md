# Trade Surveillance — Market Manipulation Detection

A machine learning system for detecting suspicious trading behavior (spoofing, layering) using synthetic limit-order-book event data. Built to run locally on a standard laptop, CPU-only.

---

## Overview

Financial markets generate millions of order events daily. Manual surveillance is not feasible at that scale. This project simulates realistic trading activity, engineers behavioral features per trader, and trains classifiers to flag suspicious patterns.

This is not a stock price predictor. The focus is on **order behavior analysis** — which is closer to what compliance teams and exchanges actually work on.

**ML Task:** Binary classification (normal vs. suspicious trader-window)  
**Unit of prediction:** One trader across one time window (e.g., 60 seconds)  
**Models used:** Logistic Regression, Random Forest, Isolation Forest

---

## Project Structure

```
trade-surveillance/
│
├── data/
│   ├── raw/                  # Simulated raw LOB events
│   ├── processed/            # Cleaned event data
│   ├── features/             # Engineered feature matrix
│   └── splits/               # Train / val / test splits
│
├── models/                   # Saved model artifacts + scaler
│
├── outputs/
│   ├── metrics/              # JSON metrics per run
│   ├── plots/                # ROC curves, confusion matrices
│   ├── predictions/          # Inference output CSVs
│   └── experiments/          # Experiment log + config snapshots
│
├── logs/                     # Per-run log files
│
├── src/
│   ├── simulate_data.py
│   ├── validate_data.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── dataset_splitter.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── experiment_tracker.py
│   └── utils/
│       ├── config_loader.py
│       └── logger.py
│
├── notebooks/
│   └── exploration.ipynb     # EDA only, not part of the pipeline
│
├── tests/
│   ├── test_simulate_data.py
│   ├── test_feature_engineering.py
│   └── test_preprocess.py
│
├── config.yaml
├── requirements.txt
├── run_training.py           # Entry point: full training pipeline
├── run_inference.py          # Entry point: inference pipeline
└── README.md
```

---

## Setup

**Requirements:** Python 3.10+, CPU-only, no GPU needed.

```bash
git clone https://github.com/your-username/trade-surveillance.git
cd trade-surveillance

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Quickstart

**Run the full training pipeline:**

```bash
python run_training.py
```

This will:
1. Simulate synthetic LOB event data
2. Validate and clean the data
3. Engineer behavioral features per trader-window
4. Split into train / val / test sets
5. Train Logistic Regression, Random Forest, and Isolation Forest
6. Evaluate on the test set and save metrics + plots
7. Log the experiment to `outputs/experiments/experiment_log.csv`

**Run inference on new data:**

```bash
python run_inference.py --input data/raw/new_events.csv
```

Output saved to `outputs/predictions/predictions.csv`.

---

## Features Engineered

Each feature is computed per trader per time window:

| Feature | Description |
|---|---|
| `cancellation_rate` | Fraction of placed orders that were cancelled |
| `order_to_trade_ratio` | Orders placed per actual trade executed |
| `placement_cancel_latency` | Mean time (ms) between order placement and cancellation |
| `mid_price_distance` | Mean distance of orders from current mid-price |
| `volume_imbalance` | Imbalance between buy-side and sell-side order volume |
| `burstiness` | Coefficient of variation of inter-arrival times |

Spoofing typically shows up as high cancellation rates, low latency between place/cancel, and orders clustered near the best bid/ask.

---

## Models

| Model | Type | Notes |
|---|---|---|
| Logistic Regression | Supervised | Baseline, interpretable coefficients |
| Random Forest | Supervised | Better on nonlinear patterns, feature importance available |
| Isolation Forest | Unsupervised | Anomaly detection, useful when labels are uncertain |

Class imbalance is handled via `class_weight='balanced'` for supervised models. Classification threshold is tuned on the validation set to optimize F1.

---

## Evaluation Metrics

Given the rare-event nature of manipulation detection, accuracy alone is not meaningful.

- Precision / Recall / F1
- ROC-AUC
- Precision@Top-K (flags the K most suspicious traders — relevant for triage workflows)
- Confusion matrix
- Feature importance (Random Forest)

Outputs saved to `outputs/metrics/` and `outputs/plots/`.

---

## Configuration

All parameters are controlled via `config.yaml`. Key sections:

```yaml
random_seed: 42

simulation:
  n_traders: 500
  suspicious_fraction: 0.10

preprocessing:
  time_window_seconds: 60
  scaler: standard

training:
  models:
    - logistic_regression
    - random_forest
    - isolation_forest

evaluation:
  top_k: 20
```

See `config.yaml` for the full parameter list.

---

## Experiment Tracking

Each training run:
- Gets a unique `run_id` (timestamp-based)
- Snapshots the config to `outputs/experiments/<run_id>_config.yaml`
- Appends metrics to `outputs/experiments/experiment_log.csv`

No external tracking tools required. The CSV is readable in any spreadsheet.

---

## Reproducibility

- Single `random_seed` in config propagates to numpy, random, and all sklearn estimators
- The data simulator uses the same seed, so `lob_events.csv` is deterministic for a given config
- The scaler is fit once during training and saved as `models/scaler.pkl` — applied identically during inference
- Any run can be reproduced from its config snapshot alone

---

## Running Tests

```bash
pytest tests/
```

Covers data simulation, feature engineering, and preprocessing logic.

---

## Requirements

```
numpy==1.26.4
pandas==2.2.2
scipy==1.13.0
scikit-learn==1.4.2
imbalanced-learn==0.12.3
matplotlib==3.8.4
seaborn==0.13.2
pyyaml==6.0.1
joblib==1.4.2
tqdm==4.66.4
pytest==8.2.0
jupyter==1.0.0
```

---

## What This Project Demonstrates

- Feature engineering from event-level data (not price data)
- Handling class imbalance in rare-event detection
- Threshold tuning on validation set
- Anomaly detection alongside supervised classification
- Clean ML pipeline with separated concerns per module
- Reproducible experiments without heavy tooling

---

## Notes

- All data is synthetic. No real market data is used.
- This system runs in batch mode only — no real-time streaming.
- The isolation forest model does not use labels during training. It can be useful for surfacing anomalies in unlabeled data.
- The notebook in `notebooks/` is for EDA only and is not part of the pipeline.

---

## License

MIT