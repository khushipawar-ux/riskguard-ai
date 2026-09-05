# RiskGuard AI — Phase 3: Imbalance Strategy Comparison

**Generated:** 2026-09-05 06:12 UTC

> All strategies are evaluated on the **same held-out test set** (20% of data).
> The test set is never resampled — only the training fold is modified.
> Accuracy is excluded; PR-AUC is the primary selection metric.

## Summary Table

| Strategy | PR-AUC | Best F1 | F1 @ t=0.50 | Recall @ >=80% prec | Train samples | Train fraud |
|----------|--------|---------|-------------|---------------------|--------------|-------------|
| SMOTE oversampling ✅ **Winner** | **0.7531**| 0.8085| 0.4971| 0.7857| 250,196| 22,745 |
| random undersampling | **0.7474**| 0.7979| 0.4203| 0.7857| 4,334| 394 |
| class_weight=balanced | **0.7222**| 0.8290| 0.1046| 0.8163| 227,845| 394 |


## Winner: `SMOTE oversampling`

The winning strategy is applied to the final model training in subsequent phases.

## Strategy Notes

### class_weight=balanced
Instructs the classifier loss function to penalise fraud misclassification
proportionally to the 1:578 imbalance ratio.  No data is added or removed.
Computationally cheapest and zero risk of synthetic data artefacts.

### SMOTE oversampling
Generates synthetic fraud examples by interpolating between real fraud
neighbours in feature space.  Expands the training set. Risk: synthetic
samples may not reflect real fraud patterns.

### random undersampling
Randomly discards legit transactions until the target ratio is reached.
Shrinks the training set.  Risk: discarding real patterns from the majority
class.

## Phase 4 Implication

The winner's resampling strategy feeds directly into the XGBoost / LightGBM
hyperparameter search in Phase 4.
