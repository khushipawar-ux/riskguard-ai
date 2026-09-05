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
| V4               |        0.81786  |         14.3384  |
| V14              |        0.429508 |          7.52995 |
| V8               |        0.32614  |          5.71774 |
| V12              |        0.302997 |          5.31202 |
| V15              |        0.234235 |          4.1065  |
| V26              |        0.232121 |          4.06944 |
| Amount_scaled    |        0.202044 |          3.54216 |
| V1               |        0.198605 |          3.48186 |
| V3               |        0.193583 |          3.39382 |
| HourOfDay_scaled |        0.191824 |          3.36298 |

---

## 2. Sample Case Explanations

### Case #1 — 🚨 **FLAGGED AS HIGH RISK / FRAUD**

- **Model Fraud Probability:** `99.98%`
- **Model Base Value:** `-6.9041`

**Top Risk Drivers (Pushing Score Higher):**
  - **V14** (`val=-6.17`): +2.9135 SHAP impact (Behavioral component V14 (val=-6.17) elevated risk by 2.913 SHAP impact)
  - **V10** (`val=-4.88`): +2.1800 SHAP impact (Behavioral component V10 (val=-4.88) elevated risk by 2.180 SHAP impact)
  - **V4** (`val=2.32`): +2.1054 SHAP impact (Behavioral component V4 (val=2.32) elevated risk by 2.105 SHAP impact)
  - **V12** (`val=-4.69`): +1.8365 SHAP impact (Behavioral component V12 (val=-4.69) elevated risk by 1.836 SHAP impact)

**Top Protective Factors (Pushing Score Lower):**
  - **V2** (`val=2.46`): -0.1976 SHAP impact (Behavioral component V2 (val=2.46) reduced risk by 0.198 SHAP impact)
  - **V6** (`val=-0.95`): -0.1603 SHAP impact (Behavioral component V6 (val=-0.95) reduced risk by 0.160 SHAP impact)
  - **V8** (`val=1.17`): -0.1290 SHAP impact (Behavioral component V8 (val=1.17) reduced risk by 0.129 SHAP impact)
  - **V15** (`val=0.59`): -0.1255 SHAP impact (Behavioral component V15 (val=0.59) reduced risk by 0.125 SHAP impact)

> **Risk Analyst Summary:**
> Decision: FLAGGED FOR INVESTIGATION | Fraud Probability: 100.0%
Key risk triggers: V14 (+2.913), V10 (+2.180), V4 (+2.105), V12 (+1.836).
Top protective factor: V2.

---

### Case #2 — 🚨 **FLAGGED AS HIGH RISK / FRAUD**

- **Model Fraud Probability:** `99.97%`
- **Model Base Value:** `-6.9041`

**Top Risk Drivers (Pushing Score Higher):**
  - **V4** (`val=3.32`): +2.0506 SHAP impact (Behavioral component V4 (val=3.32) elevated risk by 2.051 SHAP impact)
  - **V12** (`val=-4.61`): +1.9278 SHAP impact (Behavioral component V12 (val=-4.61) elevated risk by 1.928 SHAP impact)
  - **V14** (`val=-9.07`): +1.7930 SHAP impact (Behavioral component V14 (val=-9.07) elevated risk by 1.793 SHAP impact)
  - **V10** (`val=-5.05`): +1.7592 SHAP impact (Behavioral component V10 (val=-5.05) elevated risk by 1.759 SHAP impact)

**Top Protective Factors (Pushing Score Lower):**
  - **V22** (`val=-8.89`): -0.3484 SHAP impact (Behavioral component V22 (val=-8.89) reduced risk by 0.348 SHAP impact)
  - **V2** (`val=12.79`): -0.2312 SHAP impact (Behavioral component V2 (val=12.79) reduced risk by 0.231 SHAP impact)
  - **V27** (`val=1.27`): -0.1111 SHAP impact (Behavioral component V27 (val=1.27) reduced risk by 0.111 SHAP impact)
  - **V6** (`val=5.76`): -0.0700 SHAP impact (Behavioral component V6 (val=5.76) reduced risk by 0.070 SHAP impact)

> **Risk Analyst Summary:**
> Decision: FLAGGED FOR INVESTIGATION | Fraud Probability: 100.0%
Key risk triggers: V4 (+2.051), V12 (+1.928), V14 (+1.793), V10 (+1.759).
Top protective factor: V22.

---

### Case #3 — 🚨 **FLAGGED AS HIGH RISK / FRAUD**

- **Model Fraud Probability:** `99.96%`
- **Model Base Value:** `-6.9041`

**Top Risk Drivers (Pushing Score Higher):**
  - **V14** (`val=-7.50`): +2.6552 SHAP impact (Behavioral component V14 (val=-7.50) elevated risk by 2.655 SHAP impact)
  - **V10** (`val=-5.52`): +2.4308 SHAP impact (Behavioral component V10 (val=-5.52) elevated risk by 2.431 SHAP impact)
  - **V4** (`val=3.98`): +2.0572 SHAP impact (Behavioral component V4 (val=3.98) elevated risk by 2.057 SHAP impact)
  - **V12** (`val=-7.13`): +1.8856 SHAP impact (Behavioral component V12 (val=-7.13) elevated risk by 1.886 SHAP impact)

**Top Protective Factors (Pushing Score Lower):**
  - **V2** (`val=2.35`): -0.2694 SHAP impact (Behavioral component V2 (val=2.35) reduced risk by 0.269 SHAP impact)
  - **V8** (`val=1.22`): -0.1903 SHAP impact (Behavioral component V8 (val=1.22) reduced risk by 0.190 SHAP impact)
  - **V6** (`val=-1.28`): -0.0976 SHAP impact (Behavioral component V6 (val=-1.28) reduced risk by 0.098 SHAP impact)
  - **V27** (`val=0.62`): -0.0903 SHAP impact (Behavioral component V27 (val=0.62) reduced risk by 0.090 SHAP impact)

> **Risk Analyst Summary:**
> Decision: FLAGGED FOR INVESTIGATION | Fraud Probability: 100.0%
Key risk triggers: V14 (+2.655), V10 (+2.431), V4 (+2.057), V12 (+1.886).
Top protective factor: V2.


---

## 3. Risk Team Decision Workflow
1. **Automated Flag:** Model scores transaction $\ge$ optimal threshold.
2. **Instant Reason Extraction:** `ShapAnalyzer` computes top risk drivers in $<5$ ms.
3. **Analyst Action:** Analyst evaluates human-readable drivers to verify patterns (e.g. extreme PCA deviations, abnormal timing) and takes action.