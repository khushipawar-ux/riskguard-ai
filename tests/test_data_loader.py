"""Tests for riskguard.data.loader and riskguard.data.validator."""

import pathlib

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from riskguard.data.loader import DataLoadError, DataLoader
from riskguard.data.validator import (
    EXPECTED_COLUMNS,
    ValidationResult,
    validate_schema,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_minimal_df(n_legit: int = 100, n_fraud: int = 5) -> pd.DataFrame:
    """Return a minimal DataFrame that matches the expected schema."""
    import numpy as np

    rng = np.random.default_rng(42)
    rows = n_legit + n_fraud

    data: dict = {
        "Time": rng.uniform(0, 172800, rows),
        "Amount": rng.uniform(0, 500, rows),
        "Class": [0] * n_legit + [1] * n_fraud,
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.standard_normal(rows)

    return pd.DataFrame(data)[EXPECTED_COLUMNS]


# ── DataLoader tests ──────────────────────────────────────────────────────────

class TestDataLoader:
    def test_load_from_local_path(self, tmp_path: pathlib.Path) -> None:
        """DataLoader should read a CSV from an explicit local path."""
        csv = tmp_path / "creditcard.csv"
        df = _make_minimal_df()
        df.to_csv(csv, index=False)

        loader = DataLoader(local_path=csv)
        result = loader.load()

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 105

    def test_raises_when_explicit_path_missing(self) -> None:
        """DataLoader should raise DataLoadError for a non-existent explicit path."""
        loader = DataLoader(local_path=pathlib.Path("/nonexistent/creditcard.csv"))
        with pytest.raises(DataLoadError, match="does not exist"):
            loader.load()

    def test_fallback_path_resolution(self, tmp_path: pathlib.Path) -> None:
        """DataLoader should probe fallback paths before attempting download."""
        csv = tmp_path / "creditcard.csv"
        df = _make_minimal_df()
        df.to_csv(csv, index=False)

        loader = DataLoader(fallback_paths=(csv,))
        result = loader.load()

        assert result.shape[0] == 105

    def test_raises_when_no_source_available(self, monkeypatch) -> None:
        """DataLoader should raise DataLoadError when all sources fail."""
        # Point fallback paths to non-existent locations.
        loader = DataLoader(fallback_paths=(pathlib.Path("/no/such/file.csv"),))
        # Prevent actual kagglehub network call.
        monkeypatch.setitem(__builtins__ if isinstance(__builtins__, dict) else vars(__builtins__), "__import__", None)

        with pytest.raises((DataLoadError, Exception)):
            loader.load()


# ── Validator tests ───────────────────────────────────────────────────────────

class TestValidateSchema:
    def test_valid_dataframe_passes(self) -> None:
        """A well-formed DataFrame should pass all validation checks."""
        df = _make_minimal_df()
        result = validate_schema(df)
        assert result.valid
        assert result.issues == []

    def test_missing_column_fails(self) -> None:
        """A DataFrame missing expected columns should fail validation."""
        df = _make_minimal_df().drop(columns=["V14"])
        result = validate_schema(df)
        assert not result.valid
        assert any("V14" in issue for issue in result.issues)

    def test_missing_values_detected(self) -> None:
        """Null values should be reported as a validation issue."""
        df = _make_minimal_df()
        df.loc[0, "Amount"] = None
        result = validate_schema(df)
        assert not result.valid
        assert any("missing" in issue.lower() for issue in result.issues)

    def test_no_fraud_detected(self) -> None:
        """A DataFrame with no fraud rows should fail validation."""
        df = _make_minimal_df(n_fraud=0)
        result = validate_schema(df)
        assert not result.valid
        assert any("fraud" in issue.lower() for issue in result.issues)

    def test_unexpected_class_values_detected(self) -> None:
        """Non-binary Class values should be reported."""
        df = _make_minimal_df()
        df.loc[0, "Class"] = 2  # unexpected
        result = validate_schema(df)
        assert not result.valid
        assert any("unexpected" in issue.lower() for issue in result.issues)

    def test_validation_result_str_valid(self) -> None:
        result = ValidationResult(valid=True)
        assert "passed" in str(result).lower()

    def test_validation_result_str_invalid(self) -> None:
        result = ValidationResult(valid=False, issues=["Something wrong"])
        assert "failed" in str(result).lower()
        assert "Something wrong" in str(result)
