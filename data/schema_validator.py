"""Validate schemas for the Home Credit Default Risk tables.

Before any feature engineering or model training, each raw CSV table is
checked against its expected schema.  Checks include:

* Required columns and their expected dtypes
* Null profiling per column (count and percentage)
* Categorical cardinality (flag columns with suspiciously high cardinality)
* Duplicate detection on primary-key columns

Results are collected into a :class:`SchemaReport` dataclass that provides
a ``passed`` boolean and human-readable ``details``.

Typical usage::

    python -m data.schema_validator            # validate all tables
    python -m data.schema_validator --table bureau  # validate one table
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pandas as pd
import structlog
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def _load_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "train.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _raw_dir(cfg: dict) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / cfg["data"]["raw_dir"]


# ---------------------------------------------------------------------------
# Expected schemas
# ---------------------------------------------------------------------------
# Maps table name -> dict of {column_name: expected_numpy_kind}.
# numpy kinds: 'i' = signed int, 'f' = float, 'O' = object/string,
# 'u' = unsigned int, 'b' = boolean.
# We use kind codes so we don't need to match exact bit-widths.

_APPLICATION_COLS: Final[dict[str, str]] = {
    "SK_ID_CURR": "i",
    "TARGET": "if",  # present in train, absent in test — allow int or float
    "NAME_CONTRACT_TYPE": "O",
    "CODE_GENDER": "O",
    "FLAG_OWN_CAR": "O",
    "FLAG_OWN_REALTY": "O",
    "CNT_CHILDREN": "if",
    "AMT_INCOME_TOTAL": "f",
    "AMT_CREDIT": "f",
    "AMT_ANNUITY": "f",
    "AMT_GOODS_PRICE": "f",
    "NAME_EDUCATION_TYPE": "O",
    "NAME_FAMILY_STATUS": "O",
    "REGION_POPULATION_RELATIVE": "f",
    "DAYS_BIRTH": "if",
    "DAYS_EMPLOYED": "if",
    "DAYS_REGISTRATION": "f",
    "DAYS_ID_PUBLISH": "if",
    "EXT_SOURCE_1": "f",
    "EXT_SOURCE_2": "f",
    "EXT_SOURCE_3": "f",
}

_BUREAU_COLS: Final[dict[str, str]] = {
    "SK_ID_CURR": "i",
    "SK_ID_BUREAU": "i",
    "CREDIT_ACTIVE": "O",
    "CREDIT_CURRENCY": "O",
    "DAYS_CREDIT": "if",
    "CREDIT_DAY_OVERDUE": "if",
    "DAYS_CREDIT_ENDDATE": "f",
    "DAYS_ENDDATE_FACT": "f",
    "AMT_CREDIT_MAX_OVERDUE": "f",
    "AMT_CREDIT_SUM": "f",
    "AMT_CREDIT_SUM_DEBT": "f",
    "AMT_CREDIT_SUM_OVERDUE": "f",
    "CREDIT_TYPE": "O",
    "DAYS_CREDIT_UPDATE": "if",
    "AMT_ANNUITY": "f",
}

_BUREAU_BALANCE_COLS: Final[dict[str, str]] = {
    "SK_ID_BUREAU": "i",
    "MONTHS_BALANCE": "if",
    "STATUS": "O",
}

_PREVIOUS_APPLICATION_COLS: Final[dict[str, str]] = {
    "SK_ID_PREV": "i",
    "SK_ID_CURR": "i",
    "NAME_CONTRACT_TYPE": "O",
    "AMT_ANNUITY": "f",
    "AMT_APPLICATION": "f",
    "AMT_CREDIT": "f",
    "AMT_GOODS_PRICE": "f",
    "NAME_CONTRACT_STATUS": "O",
    "DAYS_DECISION": "if",
    "NAME_PAYMENT_TYPE": "O",
    "CODE_REJECT_REASON": "O",
    "NAME_CLIENT_TYPE": "O",
    "NAME_GOODS_CATEGORY": "O",
    "CHANNEL_TYPE": "O",
    "NAME_SELLER_INDUSTRY": "O",
    "NAME_YIELD_GROUP": "O",
    "PRODUCT_COMBINATION": "O",
    "DAYS_FIRST_DRAWING": "f",
    "DAYS_FIRST_DUE": "f",
    "DAYS_LAST_DUE_1ST_VERSION": "f",
    "DAYS_LAST_DUE": "f",
    "DAYS_TERMINATION": "f",
}

_POS_CASH_COLS: Final[dict[str, str]] = {
    "SK_ID_PREV": "i",
    "SK_ID_CURR": "i",
    "MONTHS_BALANCE": "if",
    "CNT_INSTALMENT": "f",
    "CNT_INSTALMENT_FUTURE": "f",
    "NAME_CONTRACT_STATUS": "O",
    "SK_DPD": "if",
    "SK_DPD_DEF": "if",
}

_INSTALLMENTS_COLS: Final[dict[str, str]] = {
    "SK_ID_PREV": "i",
    "SK_ID_CURR": "i",
    "NUM_INSTALMENT_VERSION": "f",
    "NUM_INSTALMENT_NUMBER": "if",
    "DAYS_INSTALMENT": "f",
    "DAYS_ENTRY_PAYMENT": "f",
    "AMT_INSTALMENT": "f",
    "AMT_PAYMENT": "f",
}

_CREDIT_CARD_COLS: Final[dict[str, str]] = {
    "SK_ID_PREV": "i",
    "SK_ID_CURR": "i",
    "MONTHS_BALANCE": "if",
    "AMT_BALANCE": "f",
    "AMT_CREDIT_LIMIT_ACTUAL": "if",
    "AMT_DRAWINGS_ATM_CURRENT": "f",
    "AMT_DRAWINGS_CURRENT": "f",
    "AMT_DRAWINGS_OTHER_CURRENT": "f",
    "AMT_DRAWINGS_POS_CURRENT": "f",
    "AMT_INST_MIN_REGULARITY": "f",
    "AMT_PAYMENT_CURRENT": "f",
    "AMT_PAYMENT_TOTAL_CURRENT": "f",
    "AMT_RECEIVABLE_PRINCIPAL": "f",
    "AMT_RECIVABLE": "f",
    "AMT_TOTAL_RECEIVABLE": "f",
    "CNT_DRAWINGS_ATM_CURRENT": "f",
    "CNT_DRAWINGS_CURRENT": "if",
    "CNT_DRAWINGS_OTHER_CURRENT": "f",
    "CNT_DRAWINGS_POS_CURRENT": "f",
    "CNT_INSTALMENT_MATURE_CUM": "f",
    "NAME_CONTRACT_STATUS": "O",
    "SK_DPD": "if",
    "SK_DPD_DEF": "if",
}

TABLE_SCHEMAS: Final[dict[str, dict[str, str]]] = {
    "application_train": _APPLICATION_COLS,
    "application_test": {k: v for k, v in _APPLICATION_COLS.items() if k != "TARGET"},
    "bureau": _BUREAU_COLS,
    "bureau_balance": _BUREAU_BALANCE_COLS,
    "previous_application": _PREVIOUS_APPLICATION_COLS,
    "POS_CASH_balance": _POS_CASH_COLS,
    "installments_payments": _INSTALLMENTS_COLS,
    "credit_card_balance": _CREDIT_CARD_COLS,
}

# Primary keys per table (used for duplicate detection)
TABLE_PRIMARY_KEYS: Final[dict[str, list[str]]] = {
    "application_train": ["SK_ID_CURR"],
    "application_test": ["SK_ID_CURR"],
    "bureau": ["SK_ID_BUREAU"],
    "bureau_balance": ["SK_ID_BUREAU", "MONTHS_BALANCE"],
    "previous_application": ["SK_ID_PREV"],
    "POS_CASH_balance": ["SK_ID_PREV", "MONTHS_BALANCE"],
    "installments_payments": ["SK_ID_PREV", "NUM_INSTALMENT_NUMBER", "NUM_INSTALMENT_VERSION"],
    "credit_card_balance": ["SK_ID_PREV", "MONTHS_BALANCE"],
}

# Cardinality thresholds for categorical columns
MAX_CATEGORICAL_CARDINALITY: Final[int] = 500

# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class ColumnIssue:
    """A single validation issue for a column."""

    column: str
    issue_type: str  # "missing_column", "dtype_mismatch", "high_nulls", "high_cardinality", etc.
    detail: str
    severity: str = "error"  # "error" | "warning"


@dataclass
class TableReport:
    """Validation report for a single table."""

    table_name: str
    row_count: int = 0
    col_count: int = 0
    duplicate_count: int = 0
    issues: list[ColumnIssue] = field(default_factory=list)
    null_profile: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(i.severity != "error" for i in self.issues)


@dataclass
class SchemaReport:
    """Aggregated validation report across all tables."""

    table_reports: dict[str, TableReport] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.table_reports.values())

    def summary(self) -> str:
        lines: list[str] = ["=" * 60, "SCHEMA VALIDATION REPORT", "=" * 60]
        for name, report in self.table_reports.items():
            status = "PASS" if report.passed else "FAIL"
            lines.append(
                f"\n  [{status}] {name}  "
                f"({report.row_count:,} rows, {report.col_count} cols, "
                f"{report.duplicate_count} duplicates)"
            )
            for issue in report.issues:
                marker = "ERROR" if issue.severity == "error" else "WARN "
                lines.append(f"    [{marker}] {issue.issue_type}: {issue.column} — {issue.detail}")
        lines.append("\n" + "=" * 60)
        overall = "PASSED" if self.passed else "FAILED"
        lines.append(f"Overall: {overall}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _check_columns(
    df: pd.DataFrame,
    expected: dict[str, str],
    table_name: str,
) -> list[ColumnIssue]:
    """Check that required columns exist with compatible dtypes."""
    issues: list[ColumnIssue] = []
    for col, expected_kinds in expected.items():
        if col not in df.columns:
            issues.append(
                ColumnIssue(
                    column=col,
                    issue_type="missing_column",
                    detail=f"Expected column '{col}' not found in {table_name}",
                    severity="error",
                )
            )
            continue

        actual_kind = df[col].dtype.kind
        if actual_kind not in expected_kinds:
            issues.append(
                ColumnIssue(
                    column=col,
                    issue_type="dtype_mismatch",
                    detail=(
                        f"Expected dtype kind in '{expected_kinds}', "
                        f"got '{actual_kind}' ({df[col].dtype})"
                    ),
                    severity="error",
                )
            )
    return issues


def _null_profile(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Build null count and percentage per column."""
    n = len(df)
    profile: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        profile[col] = {
            "null_count": null_count,
            "null_pct": round(null_count / n * 100, 2) if n > 0 else 0.0,
        }
    return profile


def _check_nulls(
    null_prof: dict[str, dict[str, Any]],
    warn_threshold: float = 80.0,
) -> list[ColumnIssue]:
    """Flag columns where null percentage exceeds *warn_threshold*."""
    issues: list[ColumnIssue] = []
    for col, info in null_prof.items():
        if info["null_pct"] > warn_threshold:
            issues.append(
                ColumnIssue(
                    column=col,
                    issue_type="high_nulls",
                    detail=f"{info['null_pct']}% null ({info['null_count']:,} values)",
                    severity="warning",
                )
            )
    return issues


def _check_categorical_cardinality(
    df: pd.DataFrame,
    max_card: int = MAX_CATEGORICAL_CARDINALITY,
) -> list[ColumnIssue]:
    """Warn when a categorical column has too many unique values."""
    issues: list[ColumnIssue] = []
    for col in df.select_dtypes(include=["object", "category"]).columns:
        nunique = df[col].nunique()
        if nunique > max_card:
            issues.append(
                ColumnIssue(
                    column=col,
                    issue_type="high_cardinality",
                    detail=f"{nunique} unique values (threshold {max_card})",
                    severity="warning",
                )
            )
    return issues


def _check_duplicates(
    df: pd.DataFrame,
    pk_cols: list[str],
    table_name: str,
) -> tuple[int, list[ColumnIssue]]:
    """Detect duplicates on the primary key columns."""
    issues: list[ColumnIssue] = []
    # Only check columns that actually exist in the dataframe
    valid_pk = [c for c in pk_cols if c in df.columns]
    if not valid_pk:
        return 0, issues

    dup_count = int(df.duplicated(subset=valid_pk, keep="first").sum())
    if dup_count > 0:
        issues.append(
            ColumnIssue(
                column=",".join(valid_pk),
                issue_type="duplicates",
                detail=f"{dup_count:,} duplicate rows on PK {valid_pk}",
                severity="warning",
            )
        )
    return dup_count, issues


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_table(
    table_name: str,
    raw_dir: Path,
    *,
    nrows: int | None = None,
) -> TableReport:
    """Validate a single table's schema, nulls, cardinality, and duplicates.

    Parameters
    ----------
    table_name:
        Logical table name (without ``.csv`` extension).
    raw_dir:
        Directory containing raw CSV files.
    nrows:
        Optional row limit for fast validation during development.

    Returns
    -------
    TableReport
        Detailed per-table validation results.
    """
    csv_path = raw_dir / f"{table_name}.csv"
    report = TableReport(table_name=table_name)

    if not csv_path.exists():
        report.issues.append(
            ColumnIssue(
                column="(file)",
                issue_type="file_not_found",
                detail=f"File not found: {csv_path}",
                severity="error",
            )
        )
        logger.error("file_not_found", table=table_name, path=str(csv_path))
        return report

    logger.info("validating_table", table=table_name, path=str(csv_path))
    df = pd.read_csv(csv_path, nrows=nrows)

    report.row_count = len(df)
    report.col_count = len(df.columns)

    # Column and dtype check
    expected = TABLE_SCHEMAS.get(table_name, {})
    report.issues.extend(_check_columns(df, expected, table_name))

    # Null profiling
    report.null_profile = _null_profile(df)
    report.issues.extend(_check_nulls(report.null_profile))

    # Categorical cardinality
    report.issues.extend(_check_categorical_cardinality(df))

    # Duplicate detection
    pk_cols = TABLE_PRIMARY_KEYS.get(table_name, [])
    dup_count, dup_issues = _check_duplicates(df, pk_cols, table_name)
    report.duplicate_count = dup_count
    report.issues.extend(dup_issues)

    status = "passed" if report.passed else "failed"
    logger.info(
        "table_validation_complete",
        table=table_name,
        status=status,
        issues=len(report.issues),
    )
    return report


def validate_all(
    raw_dir: Path,
    *,
    nrows: int | None = None,
    tables: list[str] | None = None,
) -> SchemaReport:
    """Validate all (or selected) tables and return a combined report.

    Parameters
    ----------
    raw_dir:
        Directory containing raw CSV files.
    nrows:
        Optional row limit per table for fast validation.
    tables:
        Subset of table names to validate.  Defaults to all tables in
        :data:`TABLE_SCHEMAS`.

    Returns
    -------
    SchemaReport
        Aggregated results with a ``passed`` property and ``summary()``
        text.
    """
    if tables is None:
        tables = list(TABLE_SCHEMAS.keys())

    report = SchemaReport()
    for tbl in tables:
        report.table_reports[tbl] = validate_table(tbl, raw_dir, nrows=nrows)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate Home Credit table schemas")
    parser.add_argument("--table", type=str, default=None, help="Validate a single table")
    parser.add_argument("--nrows", type=int, default=None, help="Read only N rows (fast check)")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    raw = _raw_dir(cfg)

    tables_to_check = [args.table] if args.table else None
    schema_report = validate_all(raw, nrows=args.nrows, tables=tables_to_check)
    print(schema_report.summary())

    sys.exit(0 if schema_report.passed else 1)
