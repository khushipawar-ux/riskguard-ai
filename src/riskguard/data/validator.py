"""
riskguard.data.validator
~~~~~~~~~~~~~~~~~~~~~~~~
Schema and integrity validation for the Credit Card Fraud dataset.

Keeps validation logic separate from loading and analysis so each can
be tested and changed independently.
"""

from dataclasses import dataclass, field

import pandas as pd

from riskguard.utils.logging import get_logger

logger = get_logger(__name__)

# Expected columns present in the mlg-ulb creditcardfraud dataset.
_V_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)]
EXPECTED_COLUMNS: list[str] = ["Time"] + _V_FEATURES + ["Amount", "Class"]
TARGET_COLUMN: str = "Class"


@dataclass
class ValidationResult:
    """Outcome of a dataset validation pass.

    Attributes:
        valid:  ``True`` when the dataset passes all checks.
        issues: Human-readable descriptions of every problem found.
    """

    valid: bool
    issues: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.valid:
            return "Dataset validation passed."
        joined = "\n  - ".join(self.issues)
        return f"Dataset validation FAILED:\n  - {joined}"


def validate_schema(df: pd.DataFrame) -> ValidationResult:
    """Validate *df* against the expected Credit Card Fraud dataset schema.

    Checks:
    * All 31 expected columns are present.
    * ``Class`` column contains only 0 / 1 values.
    * No missing values (the original dataset is clean).
    * At least one fraud row exists (sanity guard against an empty slice).

    Args:
        df: Raw :class:`pandas.DataFrame` returned by :class:`DataLoader`.

    Returns:
        :class:`ValidationResult` with ``valid=True`` when all checks pass.
    """
    issues: list[str] = []

    # 1. Column presence
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    # 2. Target values
    if TARGET_COLUMN in df.columns:
        unexpected_values = set(df[TARGET_COLUMN].unique()) - {0, 1}
        if unexpected_values:
            issues.append(
                f"'{TARGET_COLUMN}' contains unexpected values: {unexpected_values}"
            )

        # 3. At least one fraud case
        fraud_count = int((df[TARGET_COLUMN] == 1).sum())
        if fraud_count == 0:
            issues.append("No fraud cases found (Class == 1 count is 0).")

    # 4. Missing values
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        cols_with_nulls = df.columns[df.isnull().any()].tolist()
        issues.append(
            f"{total_missing} missing value(s) found in columns: {cols_with_nulls}"
        )

    result = ValidationResult(valid=len(issues) == 0, issues=issues)
    if result.valid:
        logger.info("Schema validation passed.")
    else:
        logger.warning("%s", result)
    return result
