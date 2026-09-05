# RiskGuard AI — Phase 2 Baseline Metrics

**Model:** Logistic Regression Baseline (class_weight='balanced')
**Generated:** 2026-09-05 06:11 UTC

> ⚠️ Accuracy is intentionally excluded — on a 1:578 imbalanced dataset
> it is a misleading metric. PR-AUC is the primary optimisation target.

## Evaluation Results

| Metric | Value |
|--------|-------|
| **PR-AUC** (primary) | **0.7222** |
| F1 @ threshold 0.50 | 0.1046 |
| Best F1 | 0.8290 |
| Best-F1 threshold | 1.0000 |
| Recall @ ≥80 % precision | 0.8163 |

## Confusion Matrix (threshold = 0.50)

|  | Predicted: Legit | Predicted: Fraud |
|--|-----------------|-----------------|
| **Actual: Legit** | TN = 55,350 | FP = 1,514 |
| **Actual: Fraud** | FN = 9 | TP = 89 |

- **False Positive Rate:** 2.66% of legit transactions incorrectly flagged
- **True Positive Rate (Recall):** 90.82% of fraud caught

## Interpretation

- **PR-AUC** close to 1.0 means the model achieves high precision *and*
  high recall simultaneously across all thresholds.
- The **operating threshold** (0.50) is the default; see the
  threshold curve chart for alternative risk policies.
- **Recall @ ≥80 % precision** answers: "If we review only transactions the
  model is 80 %+ confident are fraud, how much fraud do we catch?"

## Notes

- Feature engineering: Amount + Time scaled with StandardScaler; HourOfDay added as an engineered feature.
- Next phase (Phase 3) will compare SMOTE, undersampling, and class weighting on XGBoost / LightGBM to improve PR-AUC further.
- Model feature names: ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount_scaled', 'Time_scaled', 'HourOfDay_scaled']
