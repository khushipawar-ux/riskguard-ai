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
| **XGBoostFraudModel** | 0.8233 ± 0.0345 | 0.8439 |
| **LightGBMFraudModel** | 0.5878 ± 0.1255 | 0.6811 |

---

## 2. Held-Out Test Set Performance

| Model | PR-AUC (Primary) | Best F1 | F1 @ t=0.50 | Recall @ ≥80% Prec | Confusion Matrix (t=0.50) |
|:---|:---:|:---:|:---:|:---:|:---|
| **Logistic Regression Baseline** | 0.7222 | 0.8290 | 0.1046 | 0.8163 | TN=55,350 FP=1,514 FN=9 TP=89 |
| **XGBoost Fraud Model** | 0.8420 | 0.8333 | 0.6021 | 0.8367 | TN=56,760 FP=104 FN=11 TP=87 |
| **LightGBM Fraud Model** | 0.5997 | 0.7011 | 0.5506 | 0.6224 | TN=56,783 FP=81 FN=30 TP=68 |

---

## 3. Decision Threshold & Risk Policy Comparison (`XGBoost Fraud Model`)

The default threshold of 0.50 is suboptimal under extreme class imbalance. Below is the performance under distinct business policies:

| Policy | threshold | precision | recall | f1 | total_cost | TP | FP | TN | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Standard Baseline (0.50) | 0.5 | 0.4555 | 0.8878 | 0.6021 | 8100.0 | 87.0 | 104.0 | 56760.0 | 11.0 |
| Max F1 Policy | 0.9709 | 0.8511 | 0.8163 | 0.8333 | 9350.0 | 80.0 | 14.0 | 56850.0 | 18.0 |
| Target Precision >=80% | 0.9606 | 0.8039 | 0.8367 | 0.82 | 8500.0 | 82.0 | 20.0 | 56844.0 | 16.0 |
| Target Precision >=90% | 0.995 | 0.9103 | 0.7245 | 0.8068 | 13675.0 | 71.0 | 7.0 | 56857.0 | 27.0 |
| Cost-Optimal Policy | 0.6945 | 0.5959 | 0.8878 | 0.7131 | 6975.0 | 87.0 | 59.0 | 56805.0 | 11.0 |

---

## 4. Key Takeaways
1. **Tree Models vs Linear Baseline:** Tree-based gradient boosting models capture non-linear relationships and interactions between anonymized PCA features without requiring explicit interaction term engineering.
2. **Threshold Tuning as Risk Policy:** Calibrating the operating threshold directly controls the precision/recall trade-off based on business operational constraints (e.g. analyst capacity vs fraud loss exposure).