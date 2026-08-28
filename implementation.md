# RiskGuard AI — Transaction Fraud Detection
### Track 02: AI Risk Manager | Sub-Problem: Card Fraud Detection

---

## 1. Project Overview

**Repo name:** `riskguard-ai`

**Goal:** Build a fraud detection system that doesn't just classify transactions as fraud/not-fraud, but acts like a risk-management tool — surfacing a probability score, a decision threshold tied to business cost, and human-readable reasons for every flag.

**Dataset:** Kaggle Credit Card Fraud dataset (`mlg-ulb/creditcardfraud`) as the primary target, with IEEE-CIS as a stretch goal for a larger, messier, multi-table dataset once the pipeline is proven.

**Why this matters:** Card fraud costs the industry tens of billions annually. The hard constraint isn't accuracy — it's the precision/recall trade-off under extreme class imbalance (fraud is typically <1% of transactions), plus the need for explainability so a human risk analyst can act on a flag rather than trust a black box.

---

## 2. Implementation Plan

### Phase 1 — Setup & EDA (Day 1)
- Load dataset, inspect shape, dtypes, missing values
- Confirm class imbalance (`Class` value counts, normalized)
- Visualize fraud vs. non-fraud distributions across key features
- Check for time-based patterns (fraud often clusters temporally)
- Correlation check between anonymized PCA features (`V1`–`V28`) and `Class`

### Phase 2 — Baseline Model (Day 1–2)
- Stratified train/test split (critical — random split can produce near-zero fraud in a fold)
- Logistic Regression baseline for an interpretable floor
- **Metric choice:** discard accuracy entirely. Use:
  - Precision-Recall AUC (primary)
  - F1 score
  - Recall-at-fixed-precision (business-relevant: "catch X% of fraud while only reviewing Y% of legit transactions")

### Phase 3 — Handle Class Imbalance (Day 2)
- Compare 3 approaches on the training fold only (never on test):
  - Class weighting (`class_weight='balanced'`)
  - SMOTE oversampling
  - Random undersampling
- Select whichever most improves PR-AUC without collapsing precision

### Phase 4 — Stronger Models (Day 2–3)
- XGBoost / LightGBM (industry standard for tabular fraud detection)
- Hyperparameter tuning via stratified k-fold CV
- Tune the **decision threshold** explicitly — this threshold *is* the risk policy, not an afterthought

### Phase 5 — Interpretability (Day 3)
- SHAP values for global feature importance
- Per-transaction SHAP explanations for flagged cases (sampled subset, not full test set, for speed)
- This is the key differentiator: a fraud score alone isn't actionable for a risk team — they need "why"

### Phase 6 — Packaging (Day 3–4)
- Inference function: `transaction → fraud probability → flag/allow decision → top 3 reasons`
- Optional: lightweight Streamlit/Gradio demo UI for the pitch video
- Save model artifact + preprocessing pipeline for reproducibility

### Phase 7 — Pitch Prep (Day 4)
- Compile final metrics, charts (PR curve, confusion matrix, SHAP summary plot)
- Draft and rehearse 5-minute pitch script
- Record demo clip of a transaction being scored live

---

## 3. Dataset Import

```python
import kagglehub
import pandas as pd

# Requires Kaggle API credentials (kaggle.json in ~/.kaggle/
# or KAGGLE_USERNAME / KAGGLE_KEY env vars)
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Dataset downloaded to:", path)

df = pd.read_csv(f"{path}/creditcard.csv")
print(df.shape)
print(df['Class'].value_counts(normalize=True))  # confirm imbalance
df.head()
```

**Manual fallback** (no Kaggle API set up):
1. Download `creditcard.csv` from the Kaggle dataset page
2. Place it in the project's working directory
3. `df = pd.read_csv("creditcard.csv")`

---

## 4. What Broke, and How We Got Out

> Running log — updated as the project progresses. This section is deliberately left mostly empty at kickoff; the actual entries here carry more weight for the pitch than a clean pre-written narrative.

Known failure modes to watch for on this problem (fill in real instances as they're hit):

| Issue | Symptom | Fix |
|---|---|---|
| Accuracy paradox | 99.9% accuracy, model never predicts fraud | Switch to PR-AUC / recall immediately |
| SMOTE leakage | Test scores look too good | Split first, oversample only the training fold |
| Unstratified CV | Some folds have ~0 fraud cases | Use `StratifiedKFold` |
| Slow SHAP | Explanation step hangs on full test set | Sample ~500 background reference transactions instead |
| Extreme LightGBM `scale_pos_weight` | Extreme `scale_pos_weight=577` pushes all probabilities to ~1.0, collapsing precision | Keep `scale_pos_weight=1.0` or moderate (5–10) in LightGBM and use explicit decision threshold tuning |
| Suboptimal default 0.50 threshold | Default 0.50 threshold yields low precision under extreme imbalance | Implement `ThresholdOptimizer` to calibrate threshold for Max F1 (0.971) or 80% Precision (0.961) |

---

## 5. Deliverables Checklist

- [x] EDA script + visualizations (Phase 1)
- [x] Baseline model + metrics (Phase 2)
- [x] Imbalance-handling comparison (Phase 3)
- [x] Tuned XGBoost/LightGBM model (Phase 4)
- [x] SHAP interpretability layer (Phase 5)
- [ ] Inference function / demo (Phase 6)
- [x] "What broke" log (filled in)
- [ ] 5-minute pitch script + recording (Phase 7)
