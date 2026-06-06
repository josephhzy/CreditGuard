"""Cross-validation strategies for credit scoring models.

Provides stratified, grouped-stratified, and temporal split strategies
to ensure robust evaluation while preventing applicant leakage across
folds and respecting the temporal structure of credit data.

Typical usage::

    from training.cv_pipeline import get_cv_splits

    splits = get_cv_splits(df, strategy="stratified_group", n_splits=5,
                           target_col="TARGET", group_col="SK_ID_CURR")
    for fold, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SplitInfo:
    """Metadata for a single CV fold."""

    fold: int
    n_train: int
    n_val: int
    train_positive_rate: float
    val_positive_rate: float


@dataclass
class CVReport:
    """Summary of the cross-validation splitting strategy."""

    strategy: str
    n_splits: int
    folds: list[SplitInfo]


# ---------------------------------------------------------------------------
# Split strategies
# ---------------------------------------------------------------------------


def stratified_kfold_splits(
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Basic stratified k-fold preserving class balance.

    Parameters
    ----------
    y:
        Target series.
    n_splits:
        Number of folds.
    random_state:
        Random seed for reproducibility.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        (train_indices, val_indices) for each fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    logger.info("stratified_kfold", n_splits=n_splits)
    yield from skf.split(np.zeros(len(y)), y)


def stratified_group_kfold_splits(
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Stratified group k-fold preventing applicant leakage.

    Ensures that all rows belonging to the same ``SK_ID_CURR`` stay in
    the same fold, while approximately preserving class balance.

    Parameters
    ----------
    y:
        Target series.
    groups:
        Group identifiers (e.g. ``SK_ID_CURR``).  All rows sharing a
        group value will appear in the same fold.
    n_splits:
        Number of folds.
    random_state:
        Random seed for reproducibility.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        (train_indices, val_indices) for each fold.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    logger.info("stratified_group_kfold", n_splits=n_splits, n_groups=groups.nunique())
    yield from sgkf.split(np.zeros(len(y)), y, groups=groups)


def temporal_split(
    dates: pd.Series,
    split_date: str | pd.Timestamp | None = None,
    train_frac: float = 0.8,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Temporal train/validation split: train on past, validate on future.

    When ``split_date`` is ``None``, the split point is computed as the
    ``train_frac`` quantile of the date distribution.

    Parameters
    ----------
    dates:
        Datetime series used for ordering (e.g. application date).
    split_date:
        Cutoff date.  Rows with dates strictly before this go to
        train; the rest go to validation.  Auto-detected when ``None``.
    train_frac:
        Fraction of data for training when auto-detecting the split
        date.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        A single (train_indices, val_indices) tuple.
    """
    dates = pd.to_datetime(dates)

    if split_date is None:
        sorted_dates = dates.dropna().sort_values()
        split_idx = int(len(sorted_dates) * train_frac)
        split_date = sorted_dates.iloc[split_idx]
        logger.info("temporal_split_auto", split_date=str(split_date), train_frac=train_frac)
    else:
        split_date = pd.Timestamp(split_date)
        logger.info("temporal_split_fixed", split_date=str(split_date))

    train_mask = dates < split_date
    val_mask = dates >= split_date

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            f"Temporal split produced empty set: "
            f"train={len(train_idx)}, val={len(val_idx)}, split_date={split_date}"
        )

    logger.info(
        "temporal_split_result",
        n_train=len(train_idx),
        n_val=len(val_idx),
        split_date=str(split_date),
    )
    yield (train_idx, val_idx)


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------


def get_cv_splits(
    df: pd.DataFrame,
    strategy: str = "stratified_group",
    n_splits: int = 5,
    target_col: str = "TARGET",
    group_col: str = "SK_ID_CURR",
    date_col: str | None = None,
    split_date: str | pd.Timestamp | None = None,
    random_state: int = 42,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], CVReport]:
    """Get cross-validation splits with a unified interface.

    Parameters
    ----------
    df:
        Full dataframe containing features, target, and group/date
        columns.
    strategy:
        One of ``"stratified"``, ``"stratified_group"``, ``"temporal"``.
    n_splits:
        Number of folds (ignored for temporal strategy).
    target_col:
        Name of the binary target column.
    group_col:
        Name of the group column (used for grouped CV).
    date_col:
        Name of the date column (required for temporal strategy).
    split_date:
        Explicit temporal split cutoff (optional).
    random_state:
        Random seed.

    Returns
    -------
    tuple[list[tuple[np.ndarray, np.ndarray]], CVReport]
        A list of (train_idx, val_idx) tuples and a summary report.

    Raises
    ------
    ValueError
        If an unknown strategy is requested or required columns are
        missing.
    """
    y = df[target_col]

    if strategy == "stratified":
        splits_iter = stratified_kfold_splits(y, n_splits=n_splits, random_state=random_state)
    elif strategy == "stratified_group":
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found in dataframe")
        splits_iter = stratified_group_kfold_splits(y, groups=df[group_col], n_splits=n_splits)
    elif strategy == "temporal":
        if date_col is None or date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found; required for temporal strategy")
        splits_iter = temporal_split(df[date_col], split_date=split_date)
    else:
        raise ValueError(
            f"Unknown CV strategy: '{strategy}'. Use stratified/stratified_group/temporal"
        )

    # Materialize splits and build report
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    fold_infos: list[SplitInfo] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits_iter):
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        info = SplitInfo(
            fold=fold_idx,
            n_train=len(train_idx),
            n_val=len(val_idx),
            train_positive_rate=float(y_train.mean()),
            val_positive_rate=float(y_val.mean()),
        )
        fold_infos.append(info)
        splits.append((train_idx, val_idx))

        logger.info(
            "cv_fold",
            fold=fold_idx,
            n_train=info.n_train,
            n_val=info.n_val,
            train_pos_rate=round(info.train_positive_rate, 4),
            val_pos_rate=round(info.val_positive_rate, 4),
        )

    actual_splits = len(splits)
    report = CVReport(strategy=strategy, n_splits=actual_splits, folds=fold_infos)
    logger.info("cv_splits_complete", strategy=strategy, n_folds=actual_splits)

    return splits, report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import yaml

    parser = argparse.ArgumentParser(description="Preview cross-validation splits")
    parser.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    parser.add_argument("--features", type=Path, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    features_path = args.features or Path(cfg["data"]["features_path"])
    if not features_path.exists():
        logger.error("features_not_found", path=str(features_path))
        raise SystemExit(1)

    df = pd.read_parquet(features_path)
    cv_cfg = cfg["training"]["cv"]

    splits, report = get_cv_splits(
        df,
        strategy=cv_cfg["strategy"],
        n_splits=cv_cfg["n_splits"],
        target_col=cfg["data"]["target_column"],
        group_col=cfg["data"]["id_column"],
    )

    print(f"\nCV Report: {report.strategy} with {report.n_splits} folds")
    for fold_info in report.folds:
        print(
            f"  Fold {fold_info.fold}: "
            f"train={fold_info.n_train:,} ({fold_info.train_positive_rate:.3%} +) | "
            f"val={fold_info.n_val:,} ({fold_info.val_positive_rate:.3%} +)"
        )
