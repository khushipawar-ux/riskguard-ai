"""
riskguard.inference.predictor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
End-to-end inference pipeline (Phase 6).

For a given transaction returns:
* Fraud probability score (0.0 - 1.0)
* Binary decision: flag / allow (based on tuned threshold)
* Top-3 human-readable reasons derived from SHAP values

This module is the integration point for the Streamlit / Gradio demo UI.
"""

# TODO (Phase 6): Implement FraudPredictor class.
