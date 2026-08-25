# RiskGuard AI -- Architecture

## System Overview

RiskGuard AI is a card fraud detection system built around a layered,
modular Python package (`riskguard`).  The design priorities are:

**Interpretability** over raw accuracy -- every flagged transaction
includes human-readable SHAP-derived reasons, making output actionable
for a risk analyst rather than a black-box score.

**Correct metrics** -- accuracy is discarded entirely.  The system is
evaluated on Precision-Recall AUC, F1, and Recall@fixed-precision because
the dataset has ~1:578 class imbalance.

---

## Package Structure

```
src/riskguard/
|
+-- config.py              # Settings dataclass (env-var driven)
|
+-- data/
|   +-- loader.py          # Dataset acquisition (local or kagglehub)
|   +-- validator.py       # Schema + integrity checks
|
+-- eda/
|   +-- analysis.py        # Pure statistical functions (no I/O)
|   +-- visualizer.py      # Chart rendering (consumes analysis output)
|
+-- features/
|   +-- engineering.py     # Scaling, feature transforms (Phase 2+)
|
+-- models/
|   +-- baseline.py        # Logistic Regression baseline (Phase 2)
|   +-- trainer.py         # XGBoost/LightGBM + CV (Phase 3-4)
|   +-- evaluator.py       # PR-AUC, F1, threshold tuning (Phase 2-4)
|
+-- explainability/
|   +-- shap_analyzer.py   # SHAP global + per-transaction (Phase 5)
|
+-- inference/
|   +-- predictor.py       # score + flag/allow + top-3 reasons (Phase 6)
|
+-- utils/
    +-- logging.py         # Logger factory (replaces print())
    +-- plotting.py        # Dark theme + save_figure helper
```

---

## Data Flow

```
creditcard.csv
     |
     v
DataLoader.load()
     |
     v
validate_schema()
     |
     v
eda.analysis.*()        <-- pure functions, no side effects
     |
     v
eda.visualizer.*()      <-- renders charts, writes to outputs/
     |
     v
features.engineering    <-- scaling, feature creation (Phase 2+)
     |
     v
models.trainer          <-- stratified split, CV, tuning (Phase 3-4)
     |
     v
explainability.shap     <-- SHAP values on flagged transactions (Phase 5)
     |
     v
inference.predictor     <-- score + decision + reasons (Phase 6)
```

---

## Design Decisions

### Why separate `analysis.py` from `visualizer.py`?
Pure functions in `analysis.py` are unit-testable without a display.
`visualizer.py` can be tested by checking that files are created.

### Why `config.py` over direct `os.environ` calls?
Centralises all configuration so changing a key requires editing one file.
`Settings` is also importable in tests without side effects.

### Why stub Phase 2-6 modules now?
The package import tree is stable from day one.  Future phases add
implementations without restructuring.

### Metric choice
Accuracy is meaningless at 1:578 imbalance -- a model predicting
"always legit" scores 99.83%.  PR-AUC captures the precision/recall
trade-off that matters for fraud operations.

### Decision threshold
The threshold is **not** left at the sklearn default of 0.5.  It is tuned
explicitly in Phase 4 to express a business-cost policy (e.g. "catch 90%
of fraud while reviewing at most 5% of legit transactions").

---

## Key External Dependencies

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` | Baseline model, metrics, CV |
| `xgboost` / `lightgbm` | Primary models (Phase 4) |
| `imbalanced-learn` | SMOTE / undersampling (Phase 3) |
| `shap` | Explainability (Phase 5) |
| `kagglehub` | Dataset download |
| `python-dotenv` | `.env` file loading |
| `streamlit` | Demo UI (Phase 6, optional) |
