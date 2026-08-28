"""
riskguard.ui.app
~~~~~~~~~~~~~~~~
Interactive Streamlit Web Application for RiskGuard AI (Phase 6).

Demonstrates:
1. Real-time transaction risk scoring with custom and preset scenarios.
2. Dynamic decision threshold tuning based on risk policies.
3. Top-N human-interpretable SHAP explanations for risk analysts.
4. Batch transaction triage and CSV export.
5. System metrics and comparative benchmarks.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import pandas as pd

# Add src to pythonpath
_SRC_PATH = pathlib.Path(__file__).resolve().parents[2]
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

import streamlit as st

from riskguard.config import Settings
from riskguard.data.loader import DataLoader
from riskguard.explainability.shap_analyzer import ShapAnalyzer
from riskguard.inference.predictor import FraudPredictor
from riskguard.models.trainer import DataSplitter
from riskguard.models.trees import XGBoostFraudModel


# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RiskGuard AI — Fraud Risk Manager",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Premium Fintech Dark Theme ─────────────────────────────────
st.markdown(
    """
    <style>
    .main {
        background-color: #0D1117;
        color: #E6EDF3;
    }
    .stApp {
        background-color: #0D1117;
    }
    .risk-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .badge-allow {
        background-color: #238636;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 16px;
        display: inline-block;
    }
    .badge-review {
        background-color: #D29922;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 16px;
        display: inline-block;
    }
    .badge-flag {
        background-color: #DA3633;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 16px;
        display: inline-block;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #58A6FF;
    }
    .reason-box {
        background-color: #21262D;
        border-left: 4px solid #DA3633;
        padding: 10px 15px;
        margin-top: 8px;
        border-radius: 4px;
    }
    .protective-box {
        background-color: #21262D;
        border-left: 4px solid #238636;
        padding: 10px 15px;
        margin-top: 8px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_production_pipeline():
    """Load or train the production model and SHAP explainer."""
    settings = Settings()
    phase4_path = settings.output_dir / "phase4" / "xgboost_fraud_model.joblib"

    # Load dataset for background and presets
    loader = DataLoader(
        local_path=settings.dataset_path,
        kaggle_slug=settings.kaggle_dataset_slug,
    )
    df = loader.load()

    splitter = DataSplitter(test_size=0.20, random_seed=settings.random_seed)
    X_train, X_test, y_train, y_test = splitter.split(df)

    if phase4_path.exists():
        model = XGBoostFraudModel.load(phase4_path)
    else:
        model = XGBoostFraudModel(n_estimators=120, max_depth=4, learning_rate=0.1, random_seed=42)
        model.fit(X_train, y_train)

    explainer = ShapAnalyzer(model=model, background_samples=400, random_seed=42)
    explainer.fit(X_train)

    predictor = FraudPredictor(model=model, explainer=explainer, threshold=model.optimal_threshold)
    return predictor, X_test, y_test


def main():
    # ── Header ────────────────────────────────────────────────────────────────
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        st.markdown("<h1 style='font-size: 50px; margin: 0;'>🛡️</h1>", unsafe_allow_html=True)
    with col_title:
        st.markdown(
            "<h1 style='margin-bottom: 0;'>RiskGuard AI — AI Risk Manager</h1>"
            "<p style='color: #8B949E; margin-top: 0;'>Real-Time Transaction Fraud Scoring • Cost-Sensitive Thresholds • SHAP Explainability</p>",
            unsafe_allow_html=True,
        )

    with st.spinner("Initializing RiskGuard AI production engine..."):
        predictor, X_test, y_test = load_production_pipeline()

    # ── Sidebar Configuration ─────────────────────────────────────────────────
    st.sidebar.image("https://raw.githubusercontent.com/feathericons/feather/master/icons/shield.svg", width=40)
    st.sidebar.header("⚙️ Risk Policy Settings")

    policy_choice = st.sidebar.selectbox(
        "Threshold Risk Policy",
        [
            "Optimal F1 Policy (t=0.971)",
            "Target Precision >=80% (t=0.961)",
            "Target Precision >=90% (t=0.995)",
            "Cost-Optimal Policy (t=0.695)",
            "Standard Baseline (t=0.500)",
            "Custom Threshold",
        ],
    )

    threshold_map = {
        "Optimal F1 Policy (t=0.971)": 0.9709,
        "Target Precision >=80% (t=0.961)": 0.9606,
        "Target Precision >=90% (t=0.995)": 0.9950,
        "Cost-Optimal Policy (t=0.695)": 0.6945,
        "Standard Baseline (t=0.500)": 0.5000,
    }

    if policy_choice == "Custom Threshold":
        selected_threshold = st.sidebar.slider("Custom Flag Threshold", 0.01, 0.99, float(predictor.threshold), 0.01)
    else:
        selected_threshold = threshold_map[policy_choice]

    review_threshold = st.sidebar.slider(
        "Secondary Review Threshold",
        0.01,
        selected_threshold,
        min(0.30, selected_threshold * 0.5),
        0.01,
    )
    predictor.threshold = selected_threshold
    predictor.review_threshold = review_threshold

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Active Model Specs")
    st.sidebar.markdown("- **Engine:** XGBoost Classifier (Hist Gradient Boosting)")
    st.sidebar.markdown("- **Test PR-AUC:** `0.8420` (Baseline: `0.7222`)")
    st.sidebar.markdown("- **Imbalance Handling:** Auto `scale_pos_weight`")
    st.sidebar.markdown("- **Explainability:** SHAP TreeExplainer")

    # ── Main Tabs ─────────────────────────────────────────────────────────────
    tab_single, tab_batch, tab_metrics = st.tabs(
        ["🔍 Single Transaction Scoring", "📁 Batch Risk Assessment", "📈 System Benchmark & SHAP"]
    )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: Single Transaction Scoring
    # ══════════════════════════════════════════════════════════════════════════
    with tab_single:
        st.subheader("Interactive Transaction Scoring Playground")

        scenario = st.selectbox(
            "Load Scenario Preset:",
            [
                "Scenario A: Normal Grocery Store Purchase ($35.00)",
                "Scenario B: Suspicious Midnight Withdrawal ($480.00)",
                "Scenario C: High-Risk Confirmed Fraud Sample",
                "Custom Input",
            ],
        )

        # Base default features
        base_features = {f"V{i}": 0.0 for i in range(1, 29)}
        base_features["Time"] = 45000.0
        base_features["Amount"] = 35.0

        if scenario == "Scenario A: Normal Grocery Store Purchase ($35.00)":
            base_features["Amount"] = 35.50
            base_features["Time"] = 52000.0
            base_features["V4"] = -0.4
            base_features["V14"] = 0.2
        elif scenario == "Scenario B: Suspicious Midnight Withdrawal ($480.00)":
            base_features["Amount"] = 480.00
            base_features["Time"] = 12000.0  # 3 AM
            base_features["V4"] = 2.1
            base_features["V14"] = -1.8
            base_features["V12"] = -1.2
        elif scenario == "Scenario C: High-Risk Confirmed Fraud Sample":
            # Extract an actual fraud record from test set
            y_arr = np.asarray(y_test, dtype=int)
            fraud_idx = np.where(y_arr == 1)[0][0]
            actual_fraud_row = X_test.iloc[fraud_idx].to_dict()
            base_features = actual_fraud_row

        col_inp1, col_inp2, col_inp3, col_inp4 = st.columns(4)
        with col_inp1:
            amount = st.number_input("Amount ($)", value=float(base_features.get("Amount", 50.0)), step=10.0)
        with col_inp2:
            time_val = st.number_input("Time (Seconds from start)", value=float(base_features.get("Time", 50000.0)), step=3600.0)
        with col_inp3:
            v4_val = st.number_input("V4 (Fraud Indicator)", value=float(base_features.get("V4", 0.0)), step=0.5)
        with col_inp4:
            v14_val = st.number_input("V14 (Separability Component)", value=float(base_features.get("V14", 0.0)), step=0.5)

        # Build transaction dict
        tx_dict = base_features.copy()
        tx_dict["Amount"] = amount
        tx_dict["Time"] = time_val
        tx_dict["V4"] = v4_val
        tx_dict["V14"] = v14_val

        if st.button("⚡ Score Transaction & Explain", type="primary"):
            res = predictor.predict(tx_dict)

            st.markdown("---")
            col_score, col_dec, col_latency = st.columns([1.5, 2, 1])

            with col_score:
                prob_pct = res.fraud_probability * 100
                st.metric("Fraud Risk Probability", f"{prob_pct:.2f}%")
                st.progress(res.fraud_probability)

            with col_dec:
                st.markdown("### Decision Recommendation")
                if res.decision == "FLAG_FRAUD":
                    st.markdown(f"<div class='badge-flag'>🚨 FLAG AS FRAUD (>{selected_threshold:.2f})</div>", unsafe_allow_html=True)
                elif res.decision == "MANUAL_REVIEW":
                    st.markdown(f"<div class='badge-review'>⚠️ MANUAL REVIEW REQUIRED (>{review_threshold:.2f})</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='badge-allow'>✅ ALLOW TRANSACTION (<{review_threshold:.2f})</div>", unsafe_allow_html=True)

            with col_latency:
                st.metric("Scoring Latency", f"{res.latency_ms:.1f} ms")

            # Narrative & Reason codes
            st.markdown("### 📋 Risk Analyst Reason Codes (SHAP Attribution)")
            st.info(res.narrative_summary)

            col_drivers, col_protective = st.columns(2)
            with col_drivers:
                st.markdown("#### 🚨 Top Risk Drivers (Pushed Score Up)")
                if res.top_reasons:
                    for r in res.top_reasons:
                        st.markdown(
                            f"<div class='reason-box'>"
                            f"<b>{r.feature}</b>: +{r.shap_impact:.4f} SHAP impact<br>"
                            f"<small style='color: #8B949E;'>{r.explanation}</small>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.write("No anomalous risk drivers.")

            with col_protective:
                st.markdown("#### 🛡️ Top Protective Factors (Pushed Score Down)")
                if res.protective_factors:
                    for r in res.protective_factors:
                        st.markdown(
                            f"<div class='protective-box'>"
                            f"<b>{r.feature}</b>: {r.shap_impact:.4f} SHAP impact<br>"
                            f"<small style='color: #8B949E;'>{r.explanation}</small>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.write("No protective factors.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: Batch Assessment
    # ══════════════════════════════════════════════════════════════════════════
    with tab_batch:
        st.subheader("Batch Transaction Triage")
        st.markdown("Upload a CSV batch of transactions or evaluate on the test split.")

        if st.button("📥 Load Sample Test Stream (50 transactions)"):
            sample_df = X_test.head(50)
            with st.spinner("Scoring batch..."):
                batch_res_df = predictor.predict_batch_df(sample_df)

            st.markdown("### Batch Scoring Results")
            n_flagged = (batch_res_df["Decision"] == "FLAG_FRAUD").sum()
            n_review = (batch_res_df["Decision"] == "MANUAL_REVIEW").sum()
            n_allow = (batch_res_df["Decision"] == "ALLOW").sum()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Transactions", len(batch_res_df))
            m2.metric("Flagged Fraud", int(n_flagged))
            m3.metric("Manual Review", int(n_review))
            m4.metric("Allowed", int(n_allow))

            st.dataframe(
                batch_res_df,
                use_container_width=True,
                column_config={
                    "Fraud_Probability": st.column_config.ProgressColumn(
                        "Fraud Probability",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.4f",
                    ),
                },
            )

            csv_data = batch_res_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Export Triage Assessment CSV",
                data=csv_data,
                file_name="riskguard_triage_results.csv",
                mime="text/csv",
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: System Benchmark & Global SHAP
    # ══════════════════════════════════════════════════════════════════════════
    with tab_metrics:
        st.subheader("Model Performance & Global Interpretability")

        b1, b2, b3 = st.columns(3)
        b1.metric("Logistic Regression Baseline", "0.7222 PR-AUC", "Floor Floor")
        b2.metric("LightGBM Classifier", "0.5997 PR-AUC", "-0.1225")
        b3.metric("XGBoost Champion", "0.8420 PR-AUC", "+0.1198 vs Baseline", delta_color="normal")

        st.markdown("---")
        st.markdown("### 📊 Benchmark Visualizations")
        col_img1, col_img2 = st.columns(2)

        settings = Settings()
        pr_path = settings.output_dir / "phase4" / "pr_curves_comparison.png"
        thresh_path = settings.output_dir / "phase4" / "threshold_policy_curves.png"
        shap_path = settings.output_dir / "phase5" / "shap_global_importance.png"

        if pr_path.exists():
            with col_img1:
                st.image(str(pr_path), caption="Precision-Recall Curves Comparison", use_container_width=True)
        if thresh_path.exists():
            with col_img2:
                st.image(str(thresh_path), caption="Operating Threshold vs Precision/Recall/F1", use_container_width=True)

        if shap_path.exists():
            st.markdown("### 🔍 Global SHAP Feature Importance")
            st.image(str(shap_path), caption="Mean Absolute SHAP Feature Attributions", use_container_width=True)


if __name__ == "__main__":
    main()
