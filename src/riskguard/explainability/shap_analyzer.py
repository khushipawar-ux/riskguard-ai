"""
riskguard.explainability.shap_analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SHAP-based global and per-transaction explainability (Phase 5).

Provides:
* **Global Feature Importance** — Mean absolute SHAP values across features,
  identifying key macro fraud patterns.
* **Per-Transaction Explanations** — Decomposes individual fraud predictions
  into human-readable risk drivers (pushing towards fraud) and protective
  factors (pushing towards legit).
* **High-performance sampling** — Uses background reference samples (e.g. 500)
  to ensure instant, sub-second explanations for live risk analysts.

Usage::

    from riskguard.explainability.shap_analyzer import ShapAnalyzer

    analyzer = ShapAnalyzer(model)
    analyzer.fit(X_train)

    # Global importance
    importance_df = analyzer.compute_global_importance(X_test)

    # Explain specific transaction
    explanation = analyzer.explain_transaction(X_test.iloc[[0]], top_n=3)
    print(explanation.human_readable_summary)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from riskguard.models.trees import BaseTreeFraudModel
from riskguard.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RiskFactor:
    """Individual feature contribution to a transaction fraud score."""

    feature_name: str
    feature_value: float
    shap_value: float
    contribution: str  # "INCREASES_RISK" (positive SHAP) or "DECREASES_RISK" (negative SHAP)
    description: str


@dataclass
class TransactionExplanation:
    """Human-readable explanation for a single transaction scoring event."""

    fraud_probability: float
    is_flagged: bool
    base_value: float
    risk_drivers: list[RiskFactor]
    protective_factors: list[RiskFactor]
    human_readable_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fraud_probability": round(self.fraud_probability, 4),
            "is_flagged": self.is_flagged,
            "base_value": round(self.base_value, 4),
            "risk_drivers": [
                {
                    "feature": rf.feature_name,
                    "value": round(rf.feature_value, 4),
                    "shap_impact": round(rf.shap_value, 4),
                    "description": rf.description,
                }
                for rf in self.risk_drivers
            ],
            "protective_factors": [
                {
                    "feature": rf.feature_name,
                    "value": round(rf.feature_value, 4),
                    "shap_impact": round(rf.shap_value, 4),
                    "description": rf.description,
                }
                for rf in self.protective_factors
            ],
            "summary": self.human_readable_summary,
        }


class ShapAnalyzer:
    """SHAP Explainer for tree-based fraud detection models."""

    def __init__(
        self,
        model: BaseTreeFraudModel,
        background_samples: int = 500,
        random_seed: int = 42,
    ) -> None:
        """Initialize ShapAnalyzer.

        Args:
            model: Trained :class:`~riskguard.models.trees.BaseTreeFraudModel` instance.
            background_samples: Number of background samples for TreeExplainer.
            random_seed: Seed for reproducible sampling.
        """
        self.model = model
        self.background_samples = background_samples
        self.random_seed = random_seed
        self._explainer: Any | None = None
        self._feature_names: list[str] = []
        self._base_value: float = 0.0

    def fit(self, X_background: pd.DataFrame) -> ShapAnalyzer:
        """Initialize TreeExplainer with background sample data.

        Args:
            X_background: Raw DataFrame used to create background reference distribution.

        Returns:
            ``self`` fitted analyzer.
        """
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "shap is required for ShapAnalyzer. Install it with: pip install shap"
            ) from exc

        if not self.model.is_fitted:
            raise RuntimeError("Cannot fit ShapAnalyzer on an unfitted model.")

        self._feature_names = self.model.feature_names

        # Transform background features
        X_trans = self.model.get_transformed_features(X_background)
        n_samples = min(len(X_trans), self.background_samples)

        if n_samples < len(X_trans):
            rng = np.random.RandomState(self.random_seed)
            indices = rng.choice(len(X_trans), size=n_samples, replace=False)
            bg_data = X_trans[indices]
        else:
            bg_data = X_trans

        logger.info(
            "Initializing SHAP TreeExplainer for %s with %d background samples...",
            self.model.model_name,
            len(bg_data),
        )

        estimator = self.model.estimator
        # Extract underlying LightGBM or XGBoost booster if applicable
        self._explainer = shap.TreeExplainer(
            estimator,
            data=bg_data,
            feature_perturbation="interventional",
        )

        expected = self._explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            self._base_value = float(expected[1] if len(expected) > 1 else expected[0])
        else:
            self._base_value = float(expected)

        logger.info("SHAP TreeExplainer initialized (base_value=%.4f).", self._base_value)
        return self

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute raw SHAP values for transformed features of *X*.

        Args:
            X: Raw feature DataFrame.

        Returns:
            2-D array of SHAP values, shape (n_samples, n_features).
        """
        self._check_is_fitted()
        X_trans = self.model.get_transformed_features(X)
        shap_vals = self._explainer.shap_values(X_trans)

        # In binary classification, some models return [neg_shap, pos_shap]
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            return np.asarray(shap_vals[1], dtype=float)
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3 and shap_vals.shape[-1] == 2:
            return np.asarray(shap_vals[:, :, 1], dtype=float)
        return np.asarray(shap_vals, dtype=float)

    def compute_global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute global feature importance ranking via mean absolute SHAP values.

        Args:
            X: Raw feature DataFrame (e.g. test set sample).

        Returns:
            DataFrame sorted by importance with columns ``["Feature", "Mean_Abs_SHAP", "Importance_Pct"]``.
        """
        self._check_is_fitted()
        shap_vals = self.compute_shap_values(X)
        mean_abs = np.mean(np.abs(shap_vals), axis=0)

        total_importance = np.sum(mean_abs)
        importance_pct = (mean_abs / total_importance * 100) if total_importance > 0 else np.zeros_like(mean_abs)

        df = pd.DataFrame(
            {
                "Feature": self._feature_names,
                "Mean_Abs_SHAP": mean_abs,
                "Importance_Pct": importance_pct,
            }
        ).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

        logger.info(
            "Global SHAP importance computed. Top 3 features: %s, %s, %s",
            df.iloc[0]["Feature"],
            df.iloc[1]["Feature"],
            df.iloc[2]["Feature"],
        )
        return df

    def explain_transaction(
        self,
        x_single: pd.DataFrame,
        top_n: int = 4,
    ) -> TransactionExplanation:
        """Generate human-readable risk explanation for a single transaction.

        Args:
            x_single: DataFrame containing a single transaction row.
            top_n: Number of top risk drivers and protective factors to extract.

        Returns:
            :class:`TransactionExplanation` instance.
        """
        self._check_is_fitted()
        if len(x_single) != 1:
            raise ValueError(f"explain_transaction expects exactly 1 row, got {len(x_single)}")

        prob = float(self.model.predict_proba(x_single)[0])
        is_flagged = bool(prob >= self.model.optimal_threshold)

        X_trans = self.model.get_transformed_features(x_single)[0]
        shap_vals = self.compute_shap_values(x_single)[0]

        factors: list[RiskFactor] = []
        for feat_name, feat_val, s_val in zip(self._feature_names, X_trans, shap_vals):
            direction = "INCREASES_RISK" if s_val > 0 else "DECREASES_RISK"
            desc = self._generate_feature_description(feat_name, feat_val, s_val)
            factors.append(
                RiskFactor(
                    feature_name=feat_name,
                    feature_value=float(feat_val),
                    shap_value=float(s_val),
                    contribution=direction,
                    description=desc,
                )
            )

        # Separate positive and negative risk contributors
        risk_drivers = sorted(
            [f for f in factors if f.shap_value > 0],
            key=lambda f: f.shap_value,
            reverse=True,
        )[:top_n]

        protective = sorted(
            [f for f in factors if f.shap_value < 0],
            key=lambda f: f.shap_value,  # lowest negative = most protective
        )[:top_n]

        summary = self._build_narrative(prob, is_flagged, risk_drivers, protective)

        return TransactionExplanation(
            fraud_probability=prob,
            is_flagged=is_flagged,
            base_value=self._base_value,
            risk_drivers=risk_drivers,
            protective_factors=protective,
            human_readable_summary=summary,
        )

    def explain_flagged_batch(
        self,
        X_flagged: pd.DataFrame,
        top_n: int = 3,
    ) -> list[TransactionExplanation]:
        """Explain a collection of flagged transactions."""
        explanations = []
        for i in range(len(X_flagged)):
            row = X_flagged.iloc[[i]]
            explanations.append(self.explain_transaction(row, top_n=top_n))
        return explanations

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_feature_description(
        self, feat_name: str, feat_val: float, shap_val: float
    ) -> str:
        """Create a human-friendly description for feature impact."""
        impact_dir = "elevated risk by" if shap_val > 0 else "reduced risk by"
        mag = abs(shap_val)

        if "Amount" in feat_name:
            return f"Transaction amount pattern (scaled={feat_val:.2f}) {impact_dir} {mag:.3f} SHAP impact"
        elif "Hour" in feat_name or "Time" in feat_name:
            return f"Timing of transaction (scaled={feat_val:.2f}) {impact_dir} {mag:.3f} SHAP impact"
        else:
            return f"Behavioral component {feat_name} (val={feat_val:.2f}) {impact_dir} {mag:.3f} SHAP impact"

    def _build_narrative(
        self,
        prob: float,
        is_flagged: bool,
        risk_drivers: list[RiskFactor],
        protective: list[RiskFactor],
    ) -> str:
        """Create narrative summary for risk team review."""
        action = "FLAGGED FOR INVESTIGATION" if is_flagged else "CLEARED / LOW RISK"
        reasons = [f"{rd.feature_name} (+{rd.shap_value:.3f})" for rd in risk_drivers]
        reason_str = ", ".join(reasons) if reasons else "No anomalous risk factors"

        return (
            f"Decision: {action} | Fraud Probability: {prob * 100:.1f}%\n"
            f"Key risk triggers: {reason_str}.\n"
            f"Top protective factor: {protective[0].feature_name if protective else 'None'}."
        )

    def _check_is_fitted(self) -> None:
        if self._explainer is None:
            raise RuntimeError("ShapAnalyzer is not fitted. Call .fit(X_background) first.")
