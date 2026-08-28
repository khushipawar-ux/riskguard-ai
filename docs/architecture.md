# RiskGuard AI — System Architecture

## System Overview

RiskGuard AI is an end-to-end card transaction fraud detection and risk management platform built around a modular Python package (`riskguard`).

### Core Design Principles:
1. **Explainability-First**: Rather than outputting an opaque binary prediction, the system decomposes every transaction score into human-interpretable risk drivers using SHAP values.
2. **Cost-Sensitive Decision Policies**: Discards arbitrary default cutoffs (0.50) in favor of calibrated operating thresholds tied to business operational constraints (Maximum F1, Target Precision $\ge 80\%$, or Expected Financial Loss).
3. **Class-Imbalance-Aware Evaluation**: Discards accuracy entirely; evaluates exclusively on Precision-Recall AUC (PR-AUC), F1 score, and Recall-at-fixed-precision under the natural 1:578 fraud ratio.
4. **End-to-End Pipeline Encapsulation**: Embeds pre-processing transformers, tree classifiers, and calibrated thresholds inside reproducible artifacts.

---

## Package Architecture

```text
src/riskguard/
├── config.py              # Environment-based configuration (dataclass)
├── data/
│   ├── loader.py          # Dataset acquisition (local or kagglehub)
│   └── validator.py       # Schema and data integrity verification
├── eda/
│   ├── analysis.py        # Pure statistical functions (no I/O)
│   └── visualizer.py      # Chart rendering and dark-theme figures
├── features/
│   └── engineering.py     # Standard scaling and HourOfDay feature engineering
├── models/
│   ├── baseline.py        # Logistic Regression baseline with balanced weighting
│   ├── imbalance.py       # Imbalance strategies (ClassWeighting, SMOTE, Undersampling)
│   ├── comparison.py      # Imbalance strategy benchmark engine
│   ├── trees.py           # XGBoost & LightGBM production models
│   ├── threshold.py       # Decision threshold optimizer & risk policy evaluator
│   ├── trainer.py         # Stratified K-Fold CV & model training
│   └── evaluator.py       # PR-AUC, F1, and Recall@Precision evaluation
├── explainability/
│   └── shap_analyzer.py   # Global and per-transaction SHAP TreeExplainer
├── inference/
│   └── predictor.py       # End-to-end scoring, risk tiers, and top-3 reason codes
├── ui/
│   └── app.py             # Interactive Streamlit Web UI application
└── utils/
    ├── logging.py         # Centralized structured logger factory
    ├── plotting.py        # Dark-theme styling and figure export helpers
    ├── metrics_reporter.py # Baseline metric reporting
    ├── comparison_reporter.py # Imbalance comparison reporting
    ├── tree_reporter.py   # Tree models & threshold reporting
    └── explainability_reporter.py # SHAP report and visualization generator
```

---

## End-to-End Data Pipeline

```text
Credit Card Transactions (Raw Stream / CSV)
      │
      ▼
DataLoader.load() ───► validate_schema()
      │
      ▼
DataSplitter.split() [Stratified 80/20 Train/Test Split]
      │
      ├───────────────────────────────────┐
      ▼                                   ▼
FraudFeatureTransformer.fit()       Stratified 5-Fold Cross Validation
      │                                   │
      ▼                                   ▼
Model Training (XGBoost / LightGBM) ◄─────┘
      │
      ▼
ThresholdOptimizer.compare_policies() [Max F1 / Precision >= 80% / Cost-Optimal]
      │
      ▼
ShapAnalyzer.fit() [Sampled Background Reference]
      │
      ▼
FraudPredictor [Scoring -> Risk Tier -> Decision -> Top 3 SHAP Reason Codes]
      │
      ├───────────────────────────────────┐
      ▼                                   ▼
CLI Execution Pipeline               Interactive Streamlit Web UI
```

---

## Key Model & Benchmark Results

| Model Architecture | Strategy / Params | 5-Fold CV PR-AUC | Test PR-AUC | Optimal Threshold | Best F1 |
|:---|:---|:---:|:---:|:---:|:---:|
| **Logistic Regression Baseline** | `class_weight='balanced'` | — | `0.7222` | `0.9423` | `0.7419` |
| **LightGBM Classifier** | `scale_pos_weight=1.0` | `0.5878 ± 0.1255` | `0.5997` | `0.9654` | `0.7011` |
| **XGBoost Fraud Model (Champion)** | `scale_pos_weight=auto` | `0.8233 ± 0.0345` | **`0.8420`** | `0.9709` | **`0.8333`** |

---

## Technology Stack

| Component | Library / Framework | Rationale |
|:---|:---|:---|
| **Data & Computation** | `pandas`, `numpy`, `scipy` | Fast array processing and data manipulation |
| **Machine Learning** | `scikit-learn`, `xgboost`, `lightgbm` | Gradient boosted trees and stratified evaluation |
| **Imbalance Handling** | `imbalanced-learn` | SMOTE oversampling and random undersampling |
| **Explainability** | `shap` | TreeExplainer for additive game-theoretic attributions |
| **Visualization** | `matplotlib`, `seaborn` | Production-ready dark-theme reporting charts |
| **Interactive UI** | `streamlit` | Triage playground and live batch scoring interface |
| **Configuration** | `python-dotenv` | Environment variable management |
| **Testing** | `pytest`, `pytest-cov` | Automated unit, regression, and integration testing |
