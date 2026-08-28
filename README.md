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
│   ├── run_baseline.py             # Phase 2: Logistic Regression baseline entry point
│   ├── run_imbalance_comparison.py # Phase 3: Imbalance strategies benchmark entry point
│   ├── run_stronger_models.py      # Phase 4: XGBoost / LightGBM CV & Threshold Tuning
│   └── run_explainability.py       # Phase 5: SHAP Global & Local Interpretability
├── tests/
│   ├── test_data_loader.py         # Tests for dataset loading and schema validation
│   ├── test_analysis.py            # Unit tests for statistical EDA functions
│   ├── test_features.py            # Feature engineering tests
│   ├── test_baseline.py            # Logistic Regression baseline tests
│   ├── test_imbalance.py           # Imbalance strategy comparison tests
│   ├── test_trees.py               # XGBoost and LightGBM tree model tests
│   ├── test_threshold.py           # Decision threshold optimizer tests
│   ├── test_trainer.py             # Stratified K-Fold CV trainer tests
│   ├── test_shap_analyzer.py       # SHAP explainer unit tests
│   └── test_evaluator.py           # Unit tests for business metrics evaluation
├── docs/
│   └── architecture.md             # System design and technical documentation
├── outputs/                        # Generated figures, reports, and model artifacts
├── .env.example                    # Sample environment configuration
├── requirements.txt                # Project dependencies
└── pyproject.toml                  # Package configuration
```

---

## 🚀 Running the Pipelines

### 1. Phase 1: Exploratory Data Analysis (EDA)
```bash
python scripts/run_eda.py
```

### 2. Phase 2: Logistic Regression Baseline
```bash
python scripts/run_baseline.py
```

### 3. Phase 3: Imbalance Strategy Comparison
```bash
python scripts/run_imbalance_comparison.py
```

### 4. Phase 4: Stronger Models & Decision Threshold Tuning
```bash
python scripts/run_stronger_models.py
```
Outputs saved in `outputs/phase4/`:
- `xgboost_fraud_model.joblib`: Serialized champion model with preprocessing pipeline
- `model_comparison.md`: 5-fold CV and test set evaluation report
- `pr_curves_comparison.png`: Precision-Recall curves (LogReg vs XGBoost vs LightGBM)
- `threshold_policy_curves.png`: Precision, Recall, and F1 across decision thresholds

### 5. Phase 5: SHAP Interpretability Layer
```bash
python scripts/run_explainability.py
```
Outputs saved in `outputs/phase5/`:
- `interpretability_report.md`: Global feature importance & sample flagged case breakdowns
- `shap_global_importance.png`: Global SHAP feature ranking bar chart
- `shap_waterfall_case_*.png`: Local risk driver waterfall charts for flagged transactions

---

## 🧪 Testing

Run complete test suite via `pytest`:

```bash
pytest
```

---

## 📋 Implementation Roadmap

- [x] **Phase 1 — Setup & EDA**: Data validation, imbalance confirmation, temporal and feature distribution profiling.
- [x] **Phase 2 — Baseline Model**: Stratified split, Logistic Regression baseline, PR-AUC / Recall@Precision metrics.
- [x] **Phase 3 — Class Imbalance Handling**: Benchmarking class weighting vs SMOTE vs undersampling.
- [x] **Phase 4 — Stronger Models & Threshold Tuning**: Tuned XGBoost/LightGBM with 5-fold Stratified CV & explicit risk policy threshold optimization.
- [x] **Phase 5 — Interpretability**: Global feature importances & per-transaction SHAP reason codes.
- [ ] **Phase 6 — Packaging & Inference**: Fast inference API (`transaction -> risk score -> action -> top 3 reasons`).
- [ ] **Phase 7 — Pitch & Demonstration**: Demo UI and final evaluation benchmarks.

---
