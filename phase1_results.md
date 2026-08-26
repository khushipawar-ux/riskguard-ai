# RiskGuard AI — Phase 1 EDA Results ✅

## Dataset at a Glance
| Property | Value |
|---|---|
| Rows | 284,807 |
| Columns | 31 |
| Missing Values | **0** (clean) |
| Memory | 70.6 MB |

## Class Imbalance
| Class | Count | % |
|---|---|---|
| Non-Fraud | 284,315 | 99.8273% |
| Fraud | 492 | **0.1727%** |
| **Ratio** | **1 : 578** | |

---

## Chart 1 — Class Imbalance
![Class Imbalance](C:\Users\KHUSI\.gemini\antigravity-ide\brain\4dd006d6-25a9-475c-923b-f8eb95880128\01_class_imbalance.png)

---

## Chart 2 — Amount & Time Distributions
![Amount and Time](C:\Users\KHUSI\.gemini\antigravity-ide\brain\4dd006d6-25a9-475c-923b-f8eb95880128\02_amount_time_distributions.png)

> **Key finding:** Fraud transactions have a notably lower median amount ($9.25) vs. non-fraud ($22.00), but a long tail up to $2,125.

---

## Chart 3 — Temporal Fraud Patterns
![Temporal Patterns](C:\Users\KHUSI\.gemini\antigravity-ide\brain\4dd006d6-25a9-475c-923b-f8eb95880128\03_temporal_patterns.png)

> **Key finding:** Peak fraud rate of **2.05%** occurs at hour 26 — roughly 2am on day 2. Fraud clusters at low-traffic hours when detection is harder.

---

## Chart 4 — Top 12 PCA Features (KDE)
![V-Feature Distributions](C:\Users\KHUSI\.gemini\antigravity-ide\brain\4dd006d6-25a9-475c-923b-f8eb95880128\04_vfeature_distributions.png)

> **Key finding:** V14, V4, V11, V12, V10 show near-zero overlap between fraud and non-fraud — extremely discriminating even before any model training.

---

## Chart 5 — Correlation Analysis
![Correlation Analysis](C:\Users\KHUSI\.gemini\antigravity-ide\brain\4dd006d6-25a9-475c-923b-f8eb95880128\05_correlation_analysis.png)

| Direction | Feature | Pearson r |
|---|---|---|
| Fraud indicator (positive) | V11 | +0.155 |
| Fraud indicator (positive) | V4 | +0.133 |
| Fraud suppressor (negative) | V17 | -0.326 |
| Fraud suppressor (negative) | V14 | -0.302 |
| Fraud suppressor (negative) | V12 | -0.261 |

---

## Chart 6 — Feature Separability Ranking
![Feature Separability](C:\Users\KHUSI\.gemini\antigravity-ide\brain\4dd006d6-25a9-475c-923b-f8eb95880128\06_feature_separability.png)

> Cohen's d proxy — measures how well each feature separates fraud from non-fraud. V14 is by far the most discriminating single feature.

---

## Key Takeaways → Phase 2 Actions

1. **Stratified split is mandatory** — 1:578 ratio means random splits can leave folds with 0 fraud cases
2. **Drop accuracy as a metric** — a model predicting "always legit" gets 99.83% accuracy
3. **Amount & Time need scaling** — V-features are already PCA-standardised
4. **Top features to watch:** V14, V17, V12, V10, V4, V11
5. **Hour-of-day** is a viable engineered feature (2.05% fraud rate at peak hour vs ~0.17% average)
6. **Baseline target:** Logistic Regression with `class_weight='balanced'`

---

## Files Produced
| File | Path |
|---|---|
| EDA Script | [`phase1_eda.py`](file:///d:/Projects/riskguard-ai/phase1_eda.py) |
| Chart 1 | `eda_outputs/01_class_imbalance.png` |
| Chart 2 | `eda_outputs/02_amount_time_distributions.png` |
| Chart 3 | `eda_outputs/03_temporal_patterns.png` |
| Chart 4 | `eda_outputs/04_vfeature_distributions.png` |
| Chart 5 | `eda_outputs/05_correlation_analysis.png` |
| Chart 6 | `eda_outputs/06_feature_separability.png` |
| EDA Summary | [`eda_outputs/eda_summary.md`](file:///d:/Projects/riskguard-ai/eda_outputs/eda_summary.md) |
| Requirements | [`requirements.txt`](file:///d:/Projects/riskguard-ai/requirements.txt) |
| README | [`README.md`](file:///d:/Projects/riskguard-ai/README.md) |
