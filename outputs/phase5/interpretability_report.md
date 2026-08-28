# Phase 5 — SHAP Interpretability & Risk Explainability

## Executive Summary
A black-box prediction is insufficient for enterprise risk operations. Risk analysts need to know **why** a transaction was flagged before initiating a block or customer outreach.

This phase implements **SHAP (SHapley Additive exPlanations)** on the `XGBoost Fraud Model` to produce:
1. **Global feature importance** across thousands of transactions.
2. **Local per-transaction breakdowns** decomposing any individual score into contributing drivers.

---

## 1. Global Feature Importance (Top 10 Drivers)

| Feature          |   Mean_Abs_SHAP |   Importance_Pct |
|:-----------------|----------------:|-----------------:|
| V4               |        0.887788 |         15.7604  |
| V14              |        0.423398 |          7.51636 |
| V12              |        0.314571 |          5.58442 |
| V8               |        0.27908  |          4.95436 |
| HourOfDay_scaled |        0.276018 |          4.90001 |
| V26              |        0.275917 |          4.89821 |
| V18              |        0.230794 |          4.09717 |
| V3               |        0.214282 |          3.80404 |
| V16              |        0.200568 |          3.56059 |
| V1               |        0.188332 |          3.34336 |

---

## 2. Sample Case Explanations

### Case #1 — 🚨 **FLAGGED AS HIGH RISK / FRAUD**

- **Model Fraud Probability:** `99.97%`
- **Model Base Value:** `-6.7307`

**Top Risk Drivers (Pushing Score Higher):**
  - **V14** (`val=-6.17`): +4.2321 SHAP impact (Behavioral component V14 (val=-6.17) elevated risk by 4.232 SHAP impact)
  - **V12** (`val=-4.69`): +2.3963 SHAP impact (Behavioral component V12 (val=-4.69) elevated risk by 2.396 SHAP impact)
  - **V10** (`val=-4.88`): +2.1342 SHAP impact (Behavioral component V10 (val=-4.88) elevated risk by 2.134 SHAP impact)
  - **V4** (`val=2.32`): +2.0361 SHAP impact (Behavioral component V4 (val=2.32) elevated risk by 2.036 SHAP impact)

**Top Protective Factors (Pushing Score Lower):**
  - **V26** (`val=0.76`): -0.2174 SHAP impact (Behavioral component V26 (val=0.76) reduced risk by 0.217 SHAP impact)
  - **V8** (`val=1.17`): -0.1760 SHAP impact (Behavioral component V8 (val=1.17) reduced risk by 0.176 SHAP impact)
  - **V2** (`val=2.46`): -0.1702 SHAP impact (Behavioral component V2 (val=2.46) reduced risk by 0.170 SHAP impact)
  - **V5** (`val=-1.37`): -0.1283 SHAP impact (Behavioral component V5 (val=-1.37) reduced risk by 0.128 SHAP impact)

> **Risk Analyst Summary:**
> Decision: FLAGGED FOR INVESTIGATION | Fraud Probability: 100.0%
Key risk triggers: V14 (+4.232), V12 (+2.396), V10 (+2.134), V4 (+2.036).
Top protective factor: V26.

---

### Case #2 — 🚨 **FLAGGED AS HIGH RISK / FRAUD**

- **Model Fraud Probability:** `99.98%`
- **Model Base Value:** `-6.7307`

**Top Risk Drivers (Pushing Score Higher):**
  - **V14** (`val=-9.07`): +3.0097 SHAP impact (Behavioral component V14 (val=-9.07) elevated risk by 3.010 SHAP impact)
  - **V4** (`val=3.32`): +2.6451 SHAP impact (Behavioral component V4 (val=3.32) elevated risk by 2.645 SHAP impact)
  - **V12** (`val=-4.61`): +1.9684 SHAP impact (Behavioral component V12 (val=-4.61) elevated risk by 1.968 SHAP impact)
  - **V10** (`val=-5.05`): +1.3102 SHAP impact (Behavioral component V10 (val=-5.05) elevated risk by 1.310 SHAP impact)

**Top Protective Factors (Pushing Score Lower):**
  - **V2** (`val=12.79`): -0.2744 SHAP impact (Behavioral component V2 (val=12.79) reduced risk by 0.274 SHAP impact)
  - **V6** (`val=5.76`): -0.2143 SHAP impact (Behavioral component V6 (val=5.76) reduced risk by 0.214 SHAP impact)
  - **V5** (`val=-4.80`): -0.2023 SHAP impact (Behavioral component V5 (val=-4.80) reduced risk by 0.202 SHAP impact)
  - **V22** (`val=-8.89`): -0.1527 SHAP impact (Behavioral component V22 (val=-8.89) reduced risk by 0.153 SHAP impact)

> **Risk Analyst Summary:**
> Decision: FLAGGED FOR INVESTIGATION | Fraud Probability: 100.0%
Key risk triggers: V14 (+3.010), V4 (+2.645), V12 (+1.968), V10 (+1.310).
Top protective factor: V2.

---

### Case #3 — 🚨 **FLAGGED AS HIGH RISK / FRAUD**

- **Model Fraud Probability:** `99.95%`
- **Model Base Value:** `-6.7307`

**Top Risk Drivers (Pushing Score Higher):**
  - **V14** (`val=-7.50`): +4.0545 SHAP impact (Behavioral component V14 (val=-7.50) elevated risk by 4.054 SHAP impact)
  - **V4** (`val=3.98`): +2.5946 SHAP impact (Behavioral component V4 (val=3.98) elevated risk by 2.595 SHAP impact)
  - **V12** (`val=-7.13`): +2.1719 SHAP impact (Behavioral component V12 (val=-7.13) elevated risk by 2.172 SHAP impact)
  - **V10** (`val=-5.52`): +1.9114 SHAP impact (Behavioral component V10 (val=-5.52) elevated risk by 1.911 SHAP impact)

**Top Protective Factors (Pushing Score Lower):**
  - **V8** (`val=1.22`): -0.2967 SHAP impact (Behavioral component V8 (val=1.22) reduced risk by 0.297 SHAP impact)
  - **V2** (`val=2.35`): -0.2587 SHAP impact (Behavioral component V2 (val=2.35) reduced risk by 0.259 SHAP impact)
  - **V26** (`val=-0.37`): -0.1275 SHAP impact (Behavioral component V26 (val=-0.37) reduced risk by 0.128 SHAP impact)
  - **V6** (`val=-1.28`): -0.1124 SHAP impact (Behavioral component V6 (val=-1.28) reduced risk by 0.112 SHAP impact)

> **Risk Analyst Summary:**
> Decision: FLAGGED FOR INVESTIGATION | Fraud Probability: 100.0%
Key risk triggers: V14 (+4.054), V4 (+2.595), V12 (+2.172), V10 (+1.911).
Top protective factor: V8.


---

## 3. Risk Team Decision Workflow
1. **Automated Flag:** Model scores transaction $\ge$ optimal threshold.
2. **Instant Reason Extraction:** `ShapAnalyzer` computes top risk drivers in $<5$ ms.
3. **Analyst Action:** Analyst evaluates human-readable drivers to verify patterns (e.g. extreme PCA deviations, abnormal timing) and takes action.