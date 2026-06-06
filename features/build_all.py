"""Feature engineering orchestrator.

Loads the joined dataset, runs each feature family module in order,
concatenates all feature DataFrames, runs quality checks, and saves
the final feature matrix to ``data/processed/features.parquet``.

Usage:
    python -m features.build_all
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import structlog

if TYPE_CHECKING:
    from collections.abc import Callable

from features.application_profile import build_application_features
from features.bureau_history import build_bureau_features
from features.payment_behaviour import build_payment_features
from features.previous_apps import build_previous_app_features
from features.quality_checks import run_quality_checks
from features.risk_ratios import build_risk_ratio_features
from features.temporal_trends import build_temporal_features
from features.utilization import build_utilization_features

log = structlog.get_logger(__name__)

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
JOINED_PATH = PROCESSED_DIR / "joined.parquet"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"

# Ordered list of (family_name, builder_function)
FEATURE_FAMILIES: list[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = [
    ("application_profile", build_application_features),
    ("bureau_history", build_bureau_features),
    ("previous_apps", build_previous_app_features),
    ("payment_behaviour", build_payment_features),
    ("utilization", build_utilization_features),
    ("risk_ratios", build_risk_ratio_features),
    ("temporal_trends", build_temporal_features),
]


def load_joined_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the joined dataset from disk.

    Parameters
    ----------
    path : Path, optional
        Override the default joined dataset location. Accepts both
        ``.parquet`` and ``.csv`` files.

    Returns
    -------
    pd.DataFrame
    """
    path = path or JOINED_PATH

    # Try parquet first, fall back to CSV
    if path.exists():
        log.info("loading joined dataset", path=str(path))
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    csv_fallback = path.with_suffix(".csv")
    if csv_fallback.exists():
        log.info("loading joined dataset (csv fallback)", path=str(csv_fallback))
        return pd.read_csv(csv_fallback)

    msg = (
        f"Joined dataset not found at {path} or {csv_fallback}. "
        "Run the data pipeline first: `make download && make validate && make join`."
    )
    log.error(msg)
    raise FileNotFoundError(msg)


def build_features(
    df: pd.DataFrame,
    *,
    run_checks: bool = True,
    target_col: str = "TARGET",
    id_col: str = "SK_ID_CURR",
) -> pd.DataFrame:
    """Run all feature families and concatenate results.

    Parameters
    ----------
    df : pd.DataFrame
        Joined dataset with one row per SK_ID_CURR.
    run_checks : bool
        Whether to run quality checks after feature construction.
    target_col : str
        Name of the target column (used for leakage checks).
    id_col : str
        Name of the applicant ID column (preserved in output).

    Returns
    -------
    pd.DataFrame
        Feature matrix with *id_col* and all engineered features.
        The target column is included if present in the input.
    """
    log.info(
        "starting feature engineering",
        n_rows=len(df),
        n_input_cols=len(df.columns),
    )

    feature_frames: list[pd.DataFrame] = []
    total_features = 0

    for family_name, builder_fn in FEATURE_FAMILIES:
        log.info("building feature family", family=family_name)
        family_features = builder_fn(df)
        n = len(family_features.columns)
        total_features += n
        log.info(
            "feature family complete",
            family=family_name,
            n_features=n,
        )
        feature_frames.append(family_features)

    # Concatenate all feature families
    all_features = pd.concat(feature_frames, axis=1)

    # Prepend ID column
    if id_col in df.columns:
        all_features.insert(0, id_col, df[id_col])

    # Append target if present (for training; absent in scoring)
    if target_col in df.columns:
        all_features[target_col] = df[target_col]

    log.info(
        "all features assembled",
        n_features=total_features,
        n_columns=len(all_features.columns),
        n_rows=len(all_features),
    )

    # ── Quality checks ──────────────────────────────────────────────
    if run_checks:
        feature_cols = [c for c in all_features.columns if c not in (id_col, target_col)]
        target_series = all_features[target_col] if target_col in all_features.columns else None
        report = run_quality_checks(all_features[feature_cols], target=target_series)
        log.info("quality check report", summary=report.summary())

        if not report.passed:
            log.warning(
                "quality checks flagged issues -- review the report above",
            )

    return all_features


def main() -> None:
    """CLI entry point: load data, build features, save to parquet."""
    structlog.configure(
        processors=[
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
    )

    log.info("CreditGuard feature engineering pipeline")

    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    df = load_joined_dataset()
    log.info("dataset loaded", shape=df.shape)

    # Build
    features = build_features(df)

    # Coerce mixed-type object columns to string so pyarrow can serialise
    for col in features.select_dtypes(include="object").columns:
        features[col] = features[col].astype(str)

    # Save
    features.to_parquet(FEATURES_PATH, index=False)
    log.info("features saved", path=str(FEATURES_PATH), shape=features.shape)

    log.info("feature engineering pipeline complete")


if __name__ == "__main__":
    main()
