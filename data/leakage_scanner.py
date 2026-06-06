"""Scan for target leakage in the Home Credit feature set.

Target leakage occurs when information about the outcome (``TARGET``)
leaks into the feature set.  Common causes in credit modelling:

1. **Post-outcome features** — columns that are only populated *after*
   a credit decision has been made (e.g. ``SK_DPD``, ``SK_DPD_DEF``,
   ``AMT_PAYMENT``).
2. **Temporal ordering violations** — features derived from events that
   occur after the application date.
3. **Suspiciously high correlations** — a single feature correlating with
   the target at r > 0.95 almost certainly indicates leakage.
4. **Aggregation of future records** — when secondary-table records
   occurring *after* the decision date are included in aggregates.

This module maintains a blacklist, runs all four checks, and returns
a :class:`LeakageReport` dataclass.

Typical usage::

    python -m data.leakage_scanner                  # scan joined train set
    python -m data.leakage_scanner --threshold 0.90 # lower correlation threshold
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pandas as pd
import structlog
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "train.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Blacklist: post-outcome features that must never be used as predictors
# ---------------------------------------------------------------------------
# These columns describe what happened *after* a credit product was issued
# or *after* the default event.  Including them leaks future information.

BLACKLIST: Final[set[str]] = {
    # ---- Days-past-due counters (only known post-disbursement) ----
    "SK_DPD",
    "SK_DPD_DEF",
    # ---- Actual payment amounts (only known post-disbursement) ----
    "AMT_PAYMENT",
    # ---- Bureau: post-decision outcome fields ----
    "DAYS_ENDDATE_FACT",  # actual end date (post-outcome)
    "CREDIT_DAY_OVERDUE",  # current overdue days in bureau record
    "AMT_CREDIT_SUM_OVERDUE",  # outstanding overdue amount
    # ---- Previous application: termination fields ----
    "DAYS_TERMINATION",  # when the previous product ended
    "DAYS_LAST_DUE",  # last due date (shifts over time)
    # ---- Credit card: balance snapshot fields ----
    "AMT_BALANCE",  # current balance (post-disbursement)
    "AMT_RECEIVABLE_PRINCIPAL",  # current receivable
    "AMT_RECIVABLE",  # current receivable (alternate spelling)
    "AMT_TOTAL_RECEIVABLE",  # total receivable
    "AMT_PAYMENT_TOTAL_CURRENT",  # total payment this period
    # ---- Installments: actual payment tracking ----
    "DAYS_ENTRY_PAYMENT",  # date the payment was entered — only known post-disbursement
}

# Columns that should be *before* the anchor to be valid features
TEMPORAL_FEATURE_COLS: Final[dict[str, str]] = {
    # feature_col -> anchor_col it must precede
    "DAYS_FIRST_DRAWING": "DAYS_DECISION",
    "DAYS_FIRST_DUE": "DAYS_DECISION",
    "DAYS_LAST_DUE_1ST_VERSION": "DAYS_DECISION",
    "DAYS_LAST_DUE": "DAYS_DECISION",
    "DAYS_TERMINATION": "DAYS_DECISION",
}


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class LeakageFinding:
    """A single leakage finding."""

    check_type: str  # "blacklist", "temporal", "aggregation", "high_correlation"
    column: str
    detail: str
    severity: str = "error"  # "error" | "warning"


@dataclass
class LeakageReport:
    """Aggregated leakage scan results."""

    findings: list[LeakageFinding] = field(default_factory=list)
    scanned_columns: int = 0
    blacklisted_found: list[str] = field(default_factory=list)
    high_correlation_features: list[tuple[str, float]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(f.severity != "error" for f in self.findings)

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warning")

    def summary(self) -> str:
        lines: list[str] = ["=" * 60, "LEAKAGE SCAN REPORT", "=" * 60]
        lines.append(f"  Columns scanned: {self.scanned_columns}")
        lines.append(f"  Errors:   {self.error_count}")
        lines.append(f"  Warnings: {self.warning_count}")

        if self.blacklisted_found:
            lines.append(f"\n  Blacklisted columns present ({len(self.blacklisted_found)}):")
            for col in sorted(self.blacklisted_found):
                lines.append(f"    - {col}")

        if self.high_correlation_features:
            lines.append("\n  Suspiciously high target correlations:")
            for col, corr in sorted(self.high_correlation_features, key=lambda x: -abs(x[1])):
                lines.append(f"    - {col}: r={corr:.4f}")

        if self.findings:
            lines.append("\n  All findings:")
            for f in self.findings:
                marker = "ERROR" if f.severity == "error" else "WARN "
                lines.append(f"    [{marker}] [{f.check_type}] {f.column}: {f.detail}")

        lines.append("\n" + "=" * 60)
        overall = "PASSED" if self.passed else "FAILED"
        lines.append(f"Overall: {overall}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_blacklist(columns: list[str]) -> list[LeakageFinding]:
    """Flag any columns in the blacklist that appear in the dataset.

    We check both raw column names and common aggregation-prefix patterns
    (e.g. ``INSTAL_AMT_PAYMENT_mean``) so that blacklisted fields caught
    even after aggregation.
    """
    findings: list[LeakageFinding] = []
    for col in columns:
        # Direct match
        if col in BLACKLIST:
            findings.append(
                LeakageFinding(
                    check_type="blacklist",
                    column=col,
                    detail=f"Post-outcome feature '{col}' must not be used as a predictor",
                    severity="error",
                )
            )
            continue

        # Aggregated match: check if any blacklisted base name is a substring
        # Pattern: PREFIX_BLACKLISTED_stat (e.g. INSTAL_AMT_PAYMENT_mean)
        col_upper = col.upper()
        for bl in BLACKLIST:
            if f"_{bl}_" in col_upper or col_upper.endswith(f"_{bl}"):
                findings.append(
                    LeakageFinding(
                        check_type="blacklist",
                        column=col,
                        detail=f"Derived from blacklisted post-outcome feature '{bl}'",
                        severity="error",
                    )
                )
                break

    return findings


def _check_temporal_ordering(df: pd.DataFrame) -> list[LeakageFinding]:
    """Validate that temporal features precede their anchor dates.

    In the Home Credit dataset, dates are encoded as negative integers
    (days before the current application).  A feature date should be
    more negative (further in the past) than the decision/application
    date.  When the feature date is *less* negative (i.e. closer to
    today), it means the event occurred *after* the decision — leakage.
    """
    findings: list[LeakageFinding] = []

    for feat_col, anchor_col in TEMPORAL_FEATURE_COLS.items():
        # Check raw columns
        if feat_col in df.columns and anchor_col in df.columns:
            # Both are negative days: feature should be <= anchor (more negative = earlier)
            # Violation: feat_col > anchor_col  (event after decision)
            mask = df[feat_col].notna() & df[anchor_col].notna()
            if mask.sum() == 0:
                continue
            violations = (df.loc[mask, feat_col] > df.loc[mask, anchor_col]).sum()
            if violations > 0:
                pct = round(violations / mask.sum() * 100, 2)
                findings.append(
                    LeakageFinding(
                        check_type="temporal",
                        column=feat_col,
                        detail=(
                            f"{violations:,} rows ({pct}%) where {feat_col} occurs "
                            f"after {anchor_col} — possible future information"
                        ),
                        severity="warning",
                    )
                )
    return findings


def _check_aggregation_leakage(
    df: pd.DataFrame,
    raw_dir: Path,
    id_col: str,
) -> list[LeakageFinding]:
    """Detect whether aggregated features include future records.

    For secondary tables with a ``MONTHS_BALANCE`` column (POS_CASH,
    credit_card_balance), values of 0 represent the current month
    (i.e. the month of the application).  Positive values would be
    *future* months and should not exist in the data.  We flag them
    as a warning if present.
    """
    findings: list[LeakageFinding] = []

    tables_with_months = {
        "POS_CASH_balance": "MONTHS_BALANCE",
        "credit_card_balance": "MONTHS_BALANCE",
        "bureau_balance": "MONTHS_BALANCE",
    }

    for table_name, month_col in tables_with_months.items():
        csv_path = raw_dir / f"{table_name}.csv"
        if not csv_path.exists():
            continue

        # Read only the month column to check for future records
        try:
            month_series = pd.read_csv(csv_path, usecols=[month_col])[month_col]
        except (ValueError, KeyError):
            continue

        # MONTHS_BALANCE should be <= 0 (0 = current month, negative = past)
        future_count = (month_series > 0).sum()
        if future_count > 0:
            findings.append(
                LeakageFinding(
                    check_type="aggregation",
                    column=f"{table_name}.{month_col}",
                    detail=(
                        f"{future_count:,} records with MONTHS_BALANCE > 0 "
                        f"(future months) in {table_name}"
                    ),
                    severity="error",
                )
            )
        else:
            logger.info(
                "temporal_aggregation_ok",
                table=table_name,
                max_month=int(month_series.max()) if len(month_series) > 0 else None,
            )

    return findings


def _check_high_correlation(
    df: pd.DataFrame,
    target_col: str,
    threshold: float = 0.95,
) -> tuple[list[LeakageFinding], list[tuple[str, float]]]:
    """Flag features with suspiciously high correlation to the target.

    A legitimate feature rarely achieves |r| > 0.95 with the target.
    Such a correlation almost always indicates leakage or a trivial
    transformation of the target.
    """
    findings: list[LeakageFinding] = []
    high_corr: list[tuple[str, float]] = []

    if target_col not in df.columns:
        return findings, high_corr

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]

    if not numeric_cols:
        return findings, high_corr

    # Compute correlation in one vectorized call
    corrs = df[numeric_cols].corrwith(df[target_col]).dropna()

    for col, r in corrs.items():
        if abs(r) >= threshold:
            high_corr.append((str(col), float(r)))
            findings.append(
                LeakageFinding(
                    check_type="high_correlation",
                    column=str(col),
                    detail=f"|r| = {abs(r):.4f} with {target_col} (threshold {threshold})",
                    severity="error",
                )
            )
        elif abs(r) >= threshold * 0.9:
            # Near-threshold: warn so humans can investigate
            findings.append(
                LeakageFinding(
                    check_type="high_correlation",
                    column=str(col),
                    detail=f"|r| = {abs(r):.4f} — approaching leakage threshold {threshold}",
                    severity="warning",
                )
            )

    return findings, high_corr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_for_leakage(
    df: pd.DataFrame | None = None,
    config_path: Path | None = None,
    *,
    correlation_threshold: float = 0.95,
) -> LeakageReport:
    """Run the full leakage scan on the joined training dataset.

    Parameters
    ----------
    df:
        The joined DataFrame.  If ``None``, the scanner will attempt to
        load ``joined.parquet`` from the processed directory.
    config_path:
        Path to ``config/train.yaml``.
    correlation_threshold:
        Correlation coefficient above which a feature is flagged as
        suspected leakage.  Default 0.95.

    Returns
    -------
    LeakageReport
        Detailed findings with ``passed`` property and ``summary()``
        text.
    """
    cfg = _load_config(config_path)
    root = _project_root()
    raw_dir = root / cfg["data"]["raw_dir"]
    processed_dir = root / cfg["data"]["processed_dir"]
    target_col: str = cfg["data"]["target_column"]
    id_col: str = cfg["data"]["id_column"]

    # Load joined dataset if not provided
    if df is None:
        parquet_path = processed_dir / "joined.parquet"
        if not parquet_path.exists():
            logger.error("joined_dataset_not_found", path=str(parquet_path))
            raise FileNotFoundError(
                f"Joined dataset not found at {parquet_path}. "
                f"Run `python -m data.join_tables` first."
            )
        logger.info("loading_joined_dataset", path=str(parquet_path))
        df = pd.read_parquet(parquet_path)

    report = LeakageReport(scanned_columns=len(df.columns))
    logger.info("leakage_scan_start", columns=len(df.columns), rows=len(df))

    # 1. Blacklist check
    blacklist_findings = _check_blacklist(df.columns.tolist())
    report.findings.extend(blacklist_findings)
    report.blacklisted_found = [f.column for f in blacklist_findings]
    logger.info("blacklist_check_done", found=len(blacklist_findings))

    # 2. Temporal ordering check
    temporal_findings = _check_temporal_ordering(df)
    report.findings.extend(temporal_findings)
    logger.info("temporal_check_done", findings=len(temporal_findings))

    # 3. Aggregation leakage check (on raw files)
    agg_findings = _check_aggregation_leakage(df, raw_dir, id_col)
    report.findings.extend(agg_findings)
    logger.info("aggregation_check_done", findings=len(agg_findings))

    # 4. High correlation check
    corr_findings, high_corr = _check_high_correlation(
        df,
        target_col,
        threshold=correlation_threshold,
    )
    report.findings.extend(corr_findings)
    report.high_correlation_features = high_corr
    logger.info("correlation_check_done", findings=len(corr_findings))

    status = "passed" if report.passed else "failed"
    logger.info(
        "leakage_scan_complete",
        status=status,
        errors=report.error_count,
        warnings=report.warning_count,
    )
    return report


def get_clean_columns(
    columns: list[str],
    *,
    target_col: str = "TARGET",
    id_col: str = "SK_ID_CURR",
) -> list[str]:
    """Return column list with blacklisted, target, and ID columns removed.

    Utility for downstream feature engineering to get a safe feature set.

    Parameters
    ----------
    columns:
        Full list of DataFrame columns.
    target_col:
        Target column name to exclude.
    id_col:
        ID column name to exclude.

    Returns
    -------
    list[str]
        Columns safe to use as predictors.
    """
    exclude = BLACKLIST | {target_col, id_col}
    clean = []
    for col in columns:
        if col in exclude:
            continue
        # Also exclude aggregated derivatives of blacklisted columns
        col_upper = col.upper()
        is_derived = any(f"_{bl}_" in col_upper or col_upper.endswith(f"_{bl}") for bl in BLACKLIST)
        if is_derived:
            continue
        clean.append(col)
    return clean


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scan for target leakage in joined dataset")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Correlation threshold for leakage flag (default: 0.95)",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    args = parser.parse_args()

    leakage_report = scan_for_leakage(
        config_path=args.config,
        correlation_threshold=args.threshold,
    )
    print(leakage_report.summary())
    sys.exit(0 if leakage_report.passed else 1)
