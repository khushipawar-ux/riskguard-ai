# RiskGuard AI -- Phase 1 EDA Summary

## Dataset
| Property | Value |
|---|---|
| Rows | 284,807 |
| Columns | 31 |
| Missing Values | 0 |

## Class Distribution
| Class | Count | Percentage |
|---|---|---|
| Non-Fraud (0) | 284,315 | 99.8273% |
| Fraud (1) | 492 | 0.1727% |
| **Imbalance Ratio** | **1 : 577** | |

## Amount Statistics
| Metric | Non-Fraud | Fraud |
|---|---|---|
| Median | $22.00 | $9.25 |
| Mean | $88.29 | $122.21 |
| Max | $25691.16 | $2125.87 |

## Temporal Insights
- Dataset spans **48.0 hours** (~2 days)
- Peak fraud rate: **2.05%** at hour index 26

## Top Discriminating Features (Pearson r with Class)
```
Positive (fraud indicator):
V19    0.034783
V21    0.040413
V2     0.091289
V4     0.133447
V11    0.154876

Negative (fraud suppressor):
V17   -0.326481
V14   -0.302544
V12   -0.260593
V10   -0.216883
V16   -0.196539
```

## Phase 2 Actions
1. Stratified train/test split -- mandatory given 1:577 imbalance
2. Discard accuracy metric -- use PR-AUC, F1, Recall@Precision
3. Scale Amount & Time (V-features already PCA-standardised)
4. Baseline: Logistic Regression with `class_weight='balanced'`
5. Candidate feature: hour-of-day (elevated fraud rates in certain hours)
