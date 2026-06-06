"""Track 4 -- Validation strategy comparison.

Compares cross-validation strategies to assess model stability:
  - Random StratifiedKFold
  - StratifiedGroupKFold (grouped by applicant or time-based proxy)
  - Temporal / Out-of-Time (OOT) split

Logs mean/std metrics and per-fold variance to MLflow.

Usage::

    python -m experiments.track_4_validation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import structlog
import yaml
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from experiments.track_1_baseline import compute_metrics

logger = structlog.get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "train.yaml"


# ---------------------------------------------------------------------------
# Cross-validation runners
# ---------------------------------------------------------------------------


def _run_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splitter: Any,
    groups: np.ndarray | None,
    seed: int,
) -> dict[str, list[float]]:
    """Train + evaluate across CV folds, returning per-fold metrics."""
    base_params: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }

    fold_metrics: dict[str, list[float]] = {
        "roc_auc": [],
        "pr_auc": [],
        "ks_statistic": [],
        "gini": [],
    }

    split_args = (X, y, groups) if groups is not None else (X, y)

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(*split_args)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**base_params)
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_va)[:, 1]

        metrics = compute_metrics(y_va, y_prob)
        for key, val in metrics.items():
            fold_metrics[key].append(val)

        logger.info("fold_complete", fold=fold_idx, **{k: round(v, 4) for k, v in metrics.items()})

    return fold_metrics


def _temporal_split(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
    split_date: str | None,
    seed: int,
) -> dict[str, list[float]]:
    """Out-of-time validation using a temporal split.

    If no explicit date column is available, uses the index order as a
    proxy for temporal ordering (common for competition datasets).
    """
    base_params: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }

    X = df.drop(columns=[target_col, id_col], errors="ignore")
    y = df[target_col].values

    # Use 80/20 temporal split based on row order
    split_point = int(len(df) * 0.8)
    X_tr, X_va = X.iloc[:split_point], X.iloc[split_point:]
    y_tr, y_va = y[:split_point], y[split_point:]

    model = lgb.LGBMClassifier(**base_params)
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_va)[:, 1]

    metrics = compute_metrics(y_va, y_prob)
    logger.info("temporal_split_complete", n_train=len(X_tr), n_val=len(X_va), **metrics)

    # Return as single-fold lists for consistent downstream handling
    return {k: [v] for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def main(config_path: Path | None = None) -> pd.DataFrame:
    """Run Track 4: validation strategy comparison.

    Parameters
    ----------
    config_path : Path, optional
        Path to ``config/train.yaml``.

    Returns
    -------
    pd.DataFrame
        Summary table of validation strategy performance with mean/std.
    """
    cfg_path = config_path or _DEFAULT_CONFIG
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    data_cfg = cfg["data"]
    cv_cfg = cfg.get("training", {}).get("cv", {})
    mlflow_cfg = cfg.get("mlflow", {})
    n_splits = cv_cfg.get("n_splits", 5)

    # -- Load features -------------------------------------------------------
    features_path = _PROJECT_ROOT / data_cfg["features_path"]
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    df = pd.read_parquet(features_path)
    target_col = data_cfg["target_column"]
    id_col = data_cfg["id_column"]

    X = df.drop(columns=[target_col, id_col], errors="ignore")
    y = df[target_col].values

    logger.info("data_loaded", n_samples=len(X), n_features=X.shape[1])

    # Group column: use SK_ID_CURR as group to prevent leakage across folds
    groups = df[id_col].values if id_col in df.columns else None

    # -- MLflow setup --------------------------------------------------------
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns/"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "creditguard") + "/track_4_validation")

    strategies: dict[str, dict[str, list[float]]] = {}

    # 1. StratifiedKFold
    logger.info("running_stratified_kfold")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    strategies["StratifiedKFold"] = _run_cv(X, y, skf, None, seed)

    # 2. StratifiedGroupKFold
    if groups is not None:
        logger.info("running_stratified_group_kfold")
        sgkf = StratifiedGroupKFold(n_splits=n_splits)
        strategies["StratifiedGroupKFold"] = _run_cv(X, y, sgkf, groups, seed)

    # 3. Temporal / OOT split
    logger.info("running_temporal_split")
    strategies["Temporal_OOT"] = _temporal_split(
        df,
        target_col,
        id_col,
        cv_cfg.get("temporal_split_date"),
        seed,
    )

    # -- Log to MLflow -------------------------------------------------------
    results: list[dict[str, Any]] = []

    for strategy_name, fold_metrics in strategies.items():
        with mlflow.start_run(run_name=strategy_name):
            mlflow.log_param("strategy", strategy_name)
            mlflow.log_param("n_splits", n_splits)
            mlflow.log_param("seed", seed)

            row: dict[str, Any] = {"strategy": strategy_name}
            for metric_name, values in fold_metrics.items():
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))
                mlflow.log_metric(f"{metric_name}_mean", round(mean_val, 6))
                mlflow.log_metric(f"{metric_name}_std", round(std_val, 6))
                row[f"{metric_name}_mean"] = mean_val
                row[f"{metric_name}_std"] = std_val

            # Log per-fold detail as artifact
            fold_df = pd.DataFrame(fold_metrics)
            fold_path = _PROJECT_ROOT / "mlflow" / f"folds_{strategy_name}.csv"
            fold_path.parent.mkdir(parents=True, exist_ok=True)
            fold_df.to_csv(fold_path, index=False)
            mlflow.log_artifact(str(fold_path))

            results.append(row)
            logger.info("strategy_complete", strategy=strategy_name)

    # -- Summary table -------------------------------------------------------
    summary = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("TRACK 4 -- VALIDATION STRATEGY COMPARISON")
    print("=" * 80)
    print(summary.to_string(index=False, float_format="%.4f"))
    print("=" * 80 + "\n")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Track 4: Validation strategy comparison")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    main(config_path=args.config)
