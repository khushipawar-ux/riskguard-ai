# RiskGuard AI — Card Transaction Fraud Detection & Risk Management

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Track 02: AI Risk Manager | Sub-Problem: Card Fraud Detection**

RiskGuard AI is a machine learning-based fraud risk detection platform that goes beyond binary classification. Designed for human risk analysts and automated triage policies, it provides:

- 📊 **Fraud Probability Scoring**: Continuous risk assessment calibrated against transaction distributions.
- 🎯 **Cost-Sensitive Decision Thresholds**: Risk policies aligned with operational costs rather than arbitrary 0.5 cutoffs.
- 🔍 **Actionable Explainability**: Top-N human-interpretable reasons for every flagged transaction using SHAP values.

---

## 🏗️ Repository Architecture

RiskGuard AI follows a modular, production-ready structure adhering to Clean Architecture principles:

```text
riskguard-ai/
├── src/
│   └── riskguard/
│       ├── config.py               # Environment-based configuration (dataclass)
│       ├── data/
│       │   ├── loader.py           # Dataset acquisition (local / kagglehub)
│       │   └── validator.py        # Schema and data integrity validation
│       ├── eda/
│       │   ├── analysis.py         # Pure statistical analysis functions
│       │   └── visualizer.py       # Publication-ready dark-theme visualizations
│       ├── features/
│       │   └── engineering.py      # Feature transforms and scaling (Phase 2+)
│       ├── models/
│       │   ├── baseline.py         # Logistic Regression baseline (Phase 2)
│       │   ├── trainer.py          # XGBoost / LightGBM CV pipeline (Phases 3-4)
│       │   └── evaluator.py        # PR-AUC, F1, Recall@Precision evaluation
│       ├── explainability/
│       │   └── shap_analyzer.py    # Global and per-transaction SHAP explainers (Phase 5)
│       ├── inference/
│       │   └── predictor.py        # End-to-end scoring and reasoning service (Phase 6)
│       └── utils/
│           ├── logging.py          # Centralized structured logger factory
│           └── plotting.py         # Shared dark-theme styling and save routines
├── scripts/
│   ├── run_eda.py                  # Phase 1: EDA Pipeline entry point
│   ├── train.py                    # Phase 2-4: Model training and evaluation
│   └── explain.py                  # Phase 5: SHAP explanation generator
├── tests/
│   ├── test_data_loader.py         # Tests for dataset loading and schema validation
│   ├── test_analysis.py            # Unit tests for pure statistical analysis functions
│   └── test_evaluator.py           # Unit tests for business metrics evaluation
├── docs/
│   └── architecture.md             # System design and technical documentation
├── outputs/                        # Generated figures, reports, and model artifacts
├── .env.example                    # Sample environment configuration
├── requirements.txt                # Project dependencies
└── pyproject.toml                  # Package configuration
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/khushipawar-ux/riskguard-ai.git
cd riskguard-ai

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and local package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 2. Configuration

Copy the example configuration file:

```bash
cp .env.example .env
```

Edit `.env` to configure paths if using an existing local dataset, or leave defaults to let `kagglehub` auto-download `creditcardfraud`.

---

## 📈 Running Phase 1: Exploratory Data Analysis (EDA)

Run the full statistical and visual EDA pipeline:

```bash
python scripts/run_eda.py
```

Outputs will be saved in `outputs/`:
- `01_class_imbalance.png`: Imbalance ratios and log counts
- `02_amount_time_distributions.png`: KDE, boxplots, and CDFs of Amount and Time
- `03_temporal_patterns.png`: Hourly volume and fraud rate patterns
- `04_vfeature_distributions.png`: KDE comparisons for top discriminating features
- `05_correlation_analysis.png`: Correlation matrix of PCA features with `Class`
- `06_feature_separability.png`: Feature ranking via Cohen's d separability score
- `eda_summary.md`: Summary metrics markdown report

---

## 🧪 Testing

Run test suite via `pytest`:

```bash
pytest
```

---

## 📋 Implementation Roadmap

- [x] **Phase 1 — Setup & EDA**: Data validation, imbalance confirmation, temporal and feature distribution profiling.
- [ ] **Phase 2 — Baseline Model**: Stratified split, Logistic Regression baseline, PR-AUC / Recall@Precision metrics.
- [ ] **Phase 3 — Class Imbalance Handling**: Benchmarking class weighting vs SMOTE vs undersampling.
- [ ] **Phase 4 — Stronger Models & Threshold Tuning**: Tuned XGBoost/LightGBM with explicit risk policy threshold optimization.
- [ ] **Phase 5 — Interpretability**: Global feature importances & per-transaction SHAP reason codes.
- [ ] **Phase 6 — Packaging & Inference**: Fast inference API (`transaction -> risk score -> action -> top 3 reasons`).
- [ ] **Phase 7 — Pitch & Demonstration**: Demo UI and final evaluation benchmarks.

---
