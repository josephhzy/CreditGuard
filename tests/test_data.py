"""Tests for the data layer: schema validation, leakage scanning, quality report, join."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.join_tables import aggregate_table
from data.leakage_scanner import (
    _check_blacklist,
    _check_high_correlation,
    get_clean_columns,
)
from data.quality_report import analyse_column
from data.schema_validator import (
    _check_columns,
    _check_duplicates,
    _null_profile,
)

# ═══════════════════════════════════════════════════════════════════════════
# Schema validator
# ═══════════════════════════════════════════════════════════════════════════


class TestSchemaValidator:
    """Tests for schema_validator helper functions."""

    def test_schema_validator_passes_valid_data(self) -> None:
        """Valid DataFrame matching the expected schema should produce no errors."""
        df = pd.DataFrame(
            {
                "SK_ID_CURR": [1, 2, 3],
                "TARGET": [0, 1, 0],
                "AMT_INCOME_TOTAL": [100_000.0, 200_000.0, 150_000.0],
            }
        )
        expected = {"SK_ID_CURR": "i", "TARGET": "if", "AMT_INCOME_TOTAL": "f"}
        issues = _check_columns(df, expected, "test_table")
        assert len(issues) == 0

    def test_schema_validator_catches_missing_column(self) -> None:
        """Missing required column should produce an error-severity issue."""
        df = pd.DataFrame({"SK_ID_CURR": [1, 2]})
        expected = {"SK_ID_CURR": "i", "TARGET": "if"}
        issues = _check_columns(df, expected, "test_table")
        missing = [i for i in issues if i.issue_type == "missing_column"]
        assert len(missing) == 1
        assert missing[0].column == "TARGET"
        assert missing[0].severity == "error"

    def test_schema_validator_catches_duplicates(self) -> None:
        """Duplicate primary-key rows should be flagged."""
        df = pd.DataFrame({"SK_ID_CURR": [1, 1, 2], "val": [10, 20, 30]})
        dup_count, issues = _check_duplicates(df, ["SK_ID_CURR"], "test_table")
        assert dup_count == 1
        assert len(issues) == 1
        assert issues[0].issue_type == "duplicates"

    def test_null_profile_computes_correctly(self) -> None:
        """Null profile should report correct counts and percentages."""
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, None, None]})
        profile = _null_profile(df)
        assert profile["a"]["null_count"] == 1
        assert profile["b"]["null_pct"] == 100.0


# ═══════════════════════════════════════════════════════════════════════════
# Leakage scanner
# ═══════════════════════════════════════════════════════════════════════════


class TestLeakageScanner:
    """Tests for leakage_scanner checks."""

    def test_leakage_scanner_detects_blacklisted(self) -> None:
        """Columns in the BLACKLIST set should produce error findings."""
        columns = ["SK_ID_CURR", "AMT_INCOME_TOTAL", "SK_DPD", "AMT_PAYMENT"]
        findings = _check_blacklist(columns)
        flagged = {f.column for f in findings}
        assert "SK_DPD" in flagged
        assert "AMT_PAYMENT" in flagged
        assert all(f.severity == "error" for f in findings)

    def test_leakage_scanner_detects_high_correlation(self) -> None:
        """A feature perfectly correlated with TARGET should be flagged."""
        rng = np.random.default_rng(42)
        n = 200
        target = rng.binomial(1, 0.5, size=n)
        df = pd.DataFrame(
            {
                "TARGET": target,
                "leaky_feature": target.astype(float) + rng.normal(0, 0.01, n),
                "safe_feature": rng.standard_normal(n),
            }
        )
        findings, high_corr = _check_high_correlation(df, "TARGET", threshold=0.95)
        leaky_cols = {col for col, _ in high_corr}
        assert "leaky_feature" in leaky_cols
        assert "safe_feature" not in leaky_cols
        # The public output of the scanner is the LeakageFinding list. Assert
        # on it directly — without this, the scanner could stop emitting
        # findings while still returning the internal correlation tuple.
        assert any(f.column == "leaky_feature" for f in findings), (
            "scanner did not emit a LeakageFinding for the leaky feature"
        )

    def test_get_clean_columns_removes_blacklisted(self) -> None:
        """get_clean_columns should exclude blacklisted, target, and ID columns."""
        columns = [
            "SK_ID_CURR",
            "TARGET",
            "AMT_INCOME_TOTAL",
            "SK_DPD",
            "INSTAL_AMT_PAYMENT_mean",
            "feat_safe",
        ]
        clean = get_clean_columns(columns)
        assert "SK_ID_CURR" not in clean
        assert "TARGET" not in clean
        assert "SK_DPD" not in clean
        assert "INSTAL_AMT_PAYMENT_mean" not in clean
        assert "AMT_INCOME_TOTAL" in clean
        assert "feat_safe" in clean


# ═══════════════════════════════════════════════════════════════════════════
# Quality report
# ═══════════════════════════════════════════════════════════════════════════


class TestQualityReport:
    """Tests for quality_report column analysis."""

    def test_quality_report_generates_report(self) -> None:
        """analyse_column should produce a valid ColumnReport for numeric data."""
        rng = np.random.default_rng(42)
        series = pd.Series(rng.standard_normal(100), name="test_col")
        report = analyse_column(series, "test_col")

        assert report.column == "test_col"
        assert report.is_numeric is True
        assert report.null_count == 0
        assert report.numeric_stats is not None
        assert report.outlier_info is not None

    def test_quality_report_categorical(self) -> None:
        """analyse_column should capture top values for categorical columns."""
        series = pd.Series(["A", "A", "B", "C", "A", None], dtype="object", name="cat")
        report = analyse_column(series, "cat")
        assert report.is_categorical is True
        assert report.top_values is not None
        assert report.null_count == 1


# ═══════════════════════════════════════════════════════════════════════════
# Join tables
# ═══════════════════════════════════════════════════════════════════════════


class TestJoinTables:
    """Tests for the table aggregation utility."""

    def test_join_tables_output_shape(self) -> None:
        """aggregate_table should reduce a many-to-one table to one row per group."""
        df = pd.DataFrame(
            {
                "SK_ID_CURR": [1, 1, 1, 2, 2],
                "AMT_CREDIT": [100.0, 200.0, 300.0, 400.0, 500.0],
                "CREDIT_TYPE": ["A", "B", "A", "C", "A"],
            }
        )
        result = aggregate_table(df, "SK_ID_CURR", "TEST")
        assert len(result) == 2
        assert "SK_ID_CURR" in result.columns
        # Should have created aggregated numeric columns
        num_cols = [c for c in result.columns if "AMT_CREDIT" in c]
        assert len(num_cols) > 0
