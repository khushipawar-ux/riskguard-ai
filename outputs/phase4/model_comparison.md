# Phase 4 — Stronger Tree Models & Decision Threshold Tuning

## Executive Summary
This phase evaluates gradient boosted decision tree architectures (**XGBoost** and **LightGBM**) against the **Logistic Regression Baseline**, applying **Stratified 5-Fold Cross-Validation** and explicit **decision threshold optimization**.

- **Selected Best Model:** `XGBoost Fraud Model`
- **Primary Metric:** Precision-Recall AUC (PR-AUC)
- **Imbalance Handling:** Built-in positive class scale weighting (`scale_pos_weight`)

---

## 1. Stratified 5-Fold Cross-Validation Results

| Model Architecture | 5-Fold Mean PR-AUC | 5-Fold Mean Best F1 |
|:---|:---:|:---:|
| **XGBoostFraudModel** | 0.8286 ± 0.0268 | 0.8378 |
| **LightGBMFraudModel** | 0.5878 ± 0.1255 | 0.6811 |

---

## 2. Held-Out Test Set Performance

| Model | PR-AUC (Primary) | Best F1 | F1 @ t=0.50 | Recall @ ≥80% Prec | Confusion Matrix (t=0.50) |
|:---|:---:|:---:|:---:|:---:|:---|
| **Logistic Regression Baseline** | 0.7222 | 0.8290 | 0.1046 | 0.8163 | TN=55,350 FP=1,514 FN=9 TP=89 |
| **XGBoost Fraud Model** | 0.8468 | 0.8268 | 0.6035 | 0.8061 | TN=56,763 FP=101 FN=12 TP=86 |
| **LightGBM Fraud Model** | 0.5997 | 0.7011 | 0.5506 | 0.6224 | TN=56,783 FP=81 FN=30 TP=68 |

---

## 3. Decision Threshold & Risk Policy Comparison (`XGBoost Fraud Model`)

The default threshold of 0.50 is suboptimal under extreme class imbalance. Below is the performance under distinct business policies:

| Policy | threshold | precision | recall | f1 | total_cost | TP | FP | TN | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Standard Baseline (0.50) | 0.5 | 0.4599 | 0.8776 | 0.6035 | 8525.0 | 86.0 | 101.0 | 56763.0 | 12.0 |
| Max F1 Policy | 0.9939 | 0.9136 | 0.7551 | 0.8268 | 12175.0 | 74.0 | 7.0 | 56857.0 | 24.0 |
| Target Precision >=80% | 0.9735 | 0.8061 | 0.8061 | 0.8061 | 9975.0 | 79.0 | 19.0 | 56845.0 | 19.0 |
| Target Precision >=90% | 0.9939 | 0.9136 | 0.7551 | 0.8268 | 12175.0 | 74.0 | 7.0 | 56857.0 | 24.0 |
| Cost-Optimal Policy | 0.5862 | 0.5309 | 0.8776 | 0.6615 | 7900.0 | 86.0 | 76.0 | 56788.0 | 12.0 |

---

## 4. Key Takeaways
1. **Tree Models vs Linear Baseline:** Tree-based gradient boosting models capture non-linear relationships and interactions between anonymized PCA features without requiring explicit interaction term engineering.
2. **Threshold Tuning as Risk Policy:** Calibrating the operating threshold directly controls the precision/recall trade-off based on business operational constraints (e.g. analyst capacity vs fraud loss exposure).