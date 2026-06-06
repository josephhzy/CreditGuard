"""Generate a comprehensive data quality report for the joined dataset.

Produces a JSON artifact containing per-column diagnostics:

* Null counts and percentages
* Unique value counts
* Numeric distribution stats (mean, std, percentiles 1/5/25/50/75/95/99)
* Categorical value distributions (top-N values with frequencies)
* Outlier detection using the IQR method

The report is saved to ``data/processed/quality_report.json`` and can be
loaded for downstream validation gates or dashboard rendering.

Typical usage::

    python -m data.quality_report                    # report on joined.parquet
    python -m data.quality_report --top-n 20         # show top-20 categories
    python -m data.quality_report --input path.parquet  # custom input
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
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
# Constants
# ---------------------------------------------------------------------------
PERCENTILES: Final[list[float]] = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
IQR_MULTIPLIER: Final[float] = 1.5
DEFAULT_TOP_N: Final[int] = 10


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NumericStats:
    """Distribution statistics for a numeric column."""

    mean: float
    std: float
    min: float
    max: float
    p01: float
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    p99: float


@dataclass
class OutlierInfo:
    """IQR-based outlier detection results."""

    lower_bound: float
    upper_bound: float
    n_outliers_low: int
    n_outliers_high: int
    pct_outliers: float


@dataclass
class ColumnReport:
    """Quality report for a single column."""

    column: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    is_numeric: bool
    is_categorical: bool
    numeric_stats: NumericStats | None = None
    outlier_info: OutlierInfo | None = None
    top_values: dict[str, int] | None = None  # value -> count


@dataclass
class QualityReport:
    """Aggregated quality report for the entire dataset."""

    dataset_path: str
    row_count: int
    col_count: int
    total_null_count: int
    total_null_pct: float
    columns: list[ColumnReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a JSON-compatible dictionary."""
        return _report_to_dict(self)


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------


def _report_to_dict(report: QualityReport) -> dict[str, Any]:
    """Convert the report to a plain dict, handling NaN and numpy types."""

    def _clean(obj: Any) -> Any:
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {_clean(k): _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    raw = asdict(report)
    return _clean(raw)


# ---------------------------------------------------------------------------
# Per-column analysis
# ---------------------------------------------------------------------------


def _analyse_numeric(series: pd.Series) -> tuple[NumericStats, OutlierInfo]:
    """Compute numeric stats and IQR-based outlier detection."""
    desc = series.describe(percentiles=PERCENTILES)

    stats = NumericStats(
        mean=float(desc.get("mean", np.nan)),
        std=float(desc.get("std", np.nan)),
        min=float(desc.get("min", np.nan)),
        max=float(desc.get("max", np.nan)),
        p01=float(desc.get("1%", np.nan)),
        p05=float(desc.get("5%", np.nan)),
        p25=float(desc.get("25%", np.nan)),
        p50=float(desc.get("50%", np.nan)),
        p75=float(desc.get("75%", np.nan)),
        p95=float(desc.get("95%", np.nan)),
        p99=float(desc.get("99%", np.nan)),
    )

    # IQR outlier detection
    q1 = stats.p25
    q3 = stats.p75
    iqr = q3 - q1
    lower = q1 - IQR_MULTIPLIER * iqr
    upper = q3 + IQR_MULTIPLIER * iqr

    non_null = series.dropna()
    n_low = int((non_null < lower).sum())
    n_high = int((non_null > upper).sum())
    n_total = len(non_null)
    pct = round((n_low + n_high) / n_total * 100, 2) if n_total > 0 else 0.0

    outliers = OutlierInfo(
        lower_bound=float(lower),
        upper_bound=float(upper),
        n_outliers_low=n_low,
        n_outliers_high=n_high,
        pct_outliers=pct,
    )
    return stats, outliers


def _analyse_categorical(series: pd.Series, top_n: int = DEFAULT_TOP_N) -> dict[str, int]:
    """Return the top-N value counts for a categorical column."""
    vc = series.value_counts(dropna=False).head(top_n)
    return {str(k): int(v) for k, v in vc.items()}


def analyse_column(
    series: pd.Series,
    col_name: str,
    top_n: int = DEFAULT_TOP_N,
) -> ColumnReport:
    """Build a :class:`ColumnReport` for a single column.

    Parameters
    ----------
    series:
        The column data.
    col_name:
        Column name.
    top_n:
        Number of top categorical values to report.

    Returns
    -------
    ColumnReport
    """
    null_count = int(series.isna().sum())
    n = len(series)
    null_pct = round(null_count / n * 100, 2) if n > 0 else 0.0
    unique_count = int(series.nunique(dropna=False))
    is_numeric = series.dtype.kind in ("i", "f", "u")
    is_categorical = series.dtype.kind == "O" or isinstance(series.dtype, pd.CategoricalDtype)

    report = ColumnReport(
        column=col_name,
        dtype=str(series.dtype),
        null_count=null_count,
        null_pct=null_pct,
        unique_count=unique_count,
        is_numeric=is_numeric,
        is_categorical=is_categorical,
    )

    if is_numeric and series.notna().sum() > 0:
        report.numeric_stats, report.outlier_info = _analyse_numeric(series)

    if is_categorical:
        report.top_values = _analyse_categorical(series, top_n=top_n)

    return report


# ---------------------------------------------------------------------------
# Full report builder
# ---------------------------------------------------------------------------


def build_quality_report(
    df: pd.DataFrame | None = None,
    config_path: Path | None = None,
    *,
    input_path: Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    save: bool = True,
) -> QualityReport:
    """Build and optionally save a data quality report.

    Parameters
    ----------
    df:
        DataFrame to analyse.  If ``None``, loads from *input_path* or
        the default ``joined.parquet``.
    config_path:
        Path to ``config/train.yaml``.
    input_path:
        Explicit path to a Parquet or CSV file to analyse.
    top_n:
        Number of top categorical values to include per column.
    save:
        When ``True``, write the report as JSON to the processed dir.

    Returns
    -------
    QualityReport
    """
    cfg = _load_config(config_path)
    root = _project_root()
    processed_dir = root / cfg["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Resolve data source
    source_label: str
    if df is not None:
        source_label = "(in-memory DataFrame)"
    else:
        if input_path is None:
            input_path = processed_dir / "joined.parquet"
        source_label = str(input_path)
        if not input_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {input_path}. Run `python -m data.join_tables` first."
            )
        logger.info("loading_dataset", path=source_label)
        if input_path.suffix == ".parquet":
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)

    assert df is not None  # for type narrowing
    n_rows, n_cols = df.shape
    total_nulls = int(df.isna().sum().sum())
    total_cells = n_rows * n_cols
    total_null_pct = round(total_nulls / total_cells * 100, 2) if total_cells > 0 else 0.0

    logger.info("building_quality_report", rows=n_rows, cols=n_cols, source=source_label)

    report = QualityReport(
        dataset_path=source_label,
        row_count=n_rows,
        col_count=n_cols,
        total_null_count=total_nulls,
        total_null_pct=total_null_pct,
    )

    for col in df.columns:
        col_report = analyse_column(df[col], col, top_n=top_n)
        report.columns.append(col_report)

    logger.info(
        "quality_report_built",
        columns_analysed=len(report.columns),
        total_nulls=total_nulls,
        total_null_pct=total_null_pct,
    )

    # Save as JSON artifact
    if save:
        out_path = processed_dir / "quality_report.json"
        with open(out_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info("quality_report_saved", path=str(out_path))

    return report


def print_summary(report: QualityReport) -> None:
    """Print a human-readable summary to stdout."""
    print("=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)
    print(f"  Source:      {report.dataset_path}")
    print(f"  Rows:        {report.row_count:,}")
    print(f"  Columns:     {report.col_count}")
    print(f"  Total nulls: {report.total_null_count:,} ({report.total_null_pct}%)")
    print()

    # Columns with highest null rates
    high_null = sorted(
        [c for c in report.columns if c.null_pct > 0],
        key=lambda c: -c.null_pct,
    )[:15]
    if high_null:
        print("  Top null columns:")
        for c in high_null:
            print(f"    {c.column:40s}  {c.null_count:>10,}  ({c.null_pct:5.1f}%)")
        print()

    # Columns with outliers
    outlier_cols = sorted(
        [c for c in report.columns if c.outlier_info and c.outlier_info.pct_outliers > 5],
        key=lambda c: -(c.outlier_info.pct_outliers if c.outlier_info else 0),
    )[:10]
    if outlier_cols:
        print("  Columns with >5% outliers (IQR method):")
        for c in outlier_cols:
            oi = c.outlier_info
            assert oi is not None
            print(
                f"    {c.column:40s}  "
                f"low={oi.n_outliers_low:>8,}  high={oi.n_outliers_high:>8,}  "
                f"({oi.pct_outliers:.1f}%)"
            )
        print()

    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate data quality report")
    parser.add_argument("--input", type=Path, default=None, help="Path to Parquet or CSV file")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Top-N categorical values")
    parser.add_argument("--no-save", action="store_true", help="Do not save JSON artifact")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    args = parser.parse_args()

    quality_report = build_quality_report(
        config_path=args.config,
        input_path=args.input,
        top_n=args.top_n,
        save=not args.no_save,
    )
    print_summary(quality_report)
