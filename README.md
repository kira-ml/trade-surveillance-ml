# trade-surveillance-ml

A local, end-to-end machine learning system for detecting market manipulation in synthetic limit-order-book data. This project covers the core ML engineering lifecycle — from data simulation and feature engineering to model training, evaluation, and inference — designed to run on a standard laptop without any distributed infrastructure.

> Built as a portfolio project to explore applied ML engineering in a financial surveillance context.

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [ML Pipeline Stages](#ml-pipeline-stages)
- [Features Engineered](#features-engineered)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Experiment Tracking](#experiment-tracking)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Skills Demonstrated](#skills-demonstrated)

---

## Project Overview

Financial markets generate millions of order events daily. Detecting manipulation patterns such as **spoofing** and **layering** — where traders place and cancel orders rapidly to move prices artificially — is a core challenge for exchanges, regulators, and compliance teams.

This project builds a trade surveillance pipeline that:

- Simulates realistic limit-order-book (LOB) events using a Poisson arrival process
- Injects synthetic manipulation patterns into a configurable fraction of traders
- Engineers behavioral features per trader from raw event data
- Trains and compares classification and anomaly detection models
- Evaluates performance with metrics appropriate for rare-event detection

The system is intentionally kept minimal and local — the focus is on **understanding the ML workflow**, not building production infrastructure.

---

## System Architecture

```
[Data Simulation] → [Validation] → [Preprocessing] → [Feature Engineering]
        → [Model Training] → [Evaluation] → [Model Persistence] → [Inference]
                     ↑                            ↑
              [Config Manager]         [Logger + Experiment Tracker]
```

- All stages are driven by a single `config.yaml`
- Each stage is an independent Python module with clearly scoped responsibilities
- Models are versioned locally using timestamped run directories
- A local `experiment_log.csv` tracks metrics across runs

---

## Project Structure

```
trade-surveillance-ml/
│
├── data/
│   ├── raw/
│   │   └── lob_events.csv
│   └── processed/
│       ├── train.csv
│       ├── validation.csv
│       ├── test.csv
│       └── scaler_params.json
│
├── models/
│   ├── run_YYYYMMDD_HHMMSS/
│   │   ├── random_forest.joblib
│   │   ├── logistic_regression.joblib
│   │   └── isolation_forest.joblib
│   └── model_registry.json
│
├── experiments/
│   └── experiment_log.csv
│
├── logs/
│   ├── pipeline.log
│   ├── roc_curve_random_forest.png
│   ├── roc_curve_logistic_regression.png
│   └── feature_importance.png
│
├── configs/
│   └── config.yaml
│
├── src/
│   ├── data/
│   │   ├── simulate.py
│   │   ├── validate.py
│   │   └── preprocess.py
│   │
│   ├── features/
│   │   └── engineer.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── persist.py
│   │
│   ├── pipeline/
│   │   └── inference.py
│   │
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       └── experiment_tracker.py
│
├── notebooks/
│   └── eda.ipynb
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ML Pipeline Stages

| Stage | Module | Description |
|---|---|---|
| Data Simulation | `simulate.py` | Generates synthetic LOB events with injected manipulation patterns |
| Data Validation | `validate.py` | Checks schema, nulls, value ranges before preprocessing |
| Preprocessing | `preprocess.py` | Cleans, normalizes, and splits data into train/val/test |
| Feature Engineering | `engineer.py` | Aggregates event-level data into trader-level behavioral features |
| Model Training | `train.py` | Trains Logistic Regression, Random Forest, Isolation Forest |
| Evaluation | `evaluate.py` | Computes metrics, ROC curves, feature importance |
| Model Persistence | `persist.py` | Saves models with local versioning via timestamped directories |
| Inference | `inference.py` | Loads persisted model and scores new trader activity |
| Experiment Tracking | `experiment_tracker.py` | Appends run metadata and metrics to local CSV log |

---

## Features Engineered

All features are computed per trader from raw event-level data:

| Feature | Description |
|---|---|
| `cancellation_rate` | Ratio of cancellations to total orders placed |
| `order_to_trade_ratio` | Total orders placed divided by executed trades |
| `avg_time_to_cancel` | Mean time (seconds) between order placement and cancellation |
| `avg_distance_from_mid` | Mean absolute difference between order price and mid-price |
| `volume_imbalance` | (buy_volume − sell_volume) / (buy_volume + sell_volume) |
| `burstiness` | Coefficient of variation of inter-arrival times per trader |

---

## Models Used

| Model | Type | Imbalance Handling |
|---|---|---|
| Logistic Regression | Supervised Classification | `class_weight=balanced` |
| Random Forest | Supervised Classification | `class_weight=balanced` |
| Isolation Forest | Unsupervised Anomaly Detection | `contamination` parameter |

---

## Evaluation Metrics

Given that manipulating traders represent a small fraction (~10%) of all traders, standard accuracy is not a suitable metric. The following are used:

- **Precision / Recall** — primary metrics for rare-event detection
- **ROC-AUC** — overall discriminative ability
- **Confusion Matrix** — breakdown of TP, FP, FN, TN
- **Precision@Top-K** — precision among the top-K highest suspicion-score traders
- **Feature Importance** — Random Forest feature contribution analysis

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/trade-surveillance-ml.git
cd trade-surveillance-ml
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python main.py
```

### Run Inference Only

```bash
python main.py --inference
```

---

## Configuration

All pipeline parameters are controlled through `configs/config.yaml`. Key settings:

```yaml
data:
  simulation:
    num_traders: 500
    manipulation_fraction: 0.10
    seed: 42

model:
  random_forest:
    n_estimators: 100
    max_depth: 10

evaluation:
  top_k: 20
```

Adjust `manipulation_fraction`, `random_seed`, and model hyperparameters to experiment with different conditions. See the full config in [`configs/config.yaml`](configs/config.yaml).

---

## Experiment Tracking

Each training run is logged to `experiments/experiment_log.csv` with:

- Run ID and timestamp
- Model name and hyperparameters
- ROC-AUC, Precision, Recall on the test set

To compare runs:

```python
from src.utils.experiment_tracker import get_best_run
best = get_best_run(metric="roc_auc")
print(best)
```

---

## Reproducibility

- All random seeds are set via `config.yaml` (`project.random_seed`)
- Each model run saves a copy of the active config inside its versioned directory
- `model_registry.json` tracks all saved runs with metadata
- Scaler parameters are persisted to `data/processed/scaler_params.json` and reloaded at inference

---

## Limitations

- Data is fully synthetic — real LOB data would require exchange APIs or licensed market data feeds
- Feature engineering is intentionally simple; real surveillance systems use much richer temporal and cross-trader signals
- No time-series modeling (e.g., LSTMs or sequence models) is used in this version
- The system is designed for learning and portfolio purposes, not production deployment

---

## Skills Demonstrated

- End-to-end ML pipeline design with clear separation of concerns
- Feature engineering from event-level time series data
- Handling class imbalance in a rare-event detection setting
- Supervised classification and unsupervised anomaly detection
- Local model versioning and experiment tracking without heavy MLOps tooling
- Configuration-driven, reproducible ML workflows

---

## Dependencies

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
imbalanced-learn==0.12.3
joblib==1.4.2
pyyaml==6.0.1
matplotlib==3.9.0
python-dateutil==2.9.0
```

---

## License

This project is intended for educational and portfolio purposes.
