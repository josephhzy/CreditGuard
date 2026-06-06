"""Track 1 -- Baseline model comparison.

Trains LogisticRegression, RandomForest, XGBoost, and LightGBM on the
same train/validation split with the same feature set. Logs all runs
to MLflow for side-by-side comparison.

Usage::

    python -m experiments.track_1_baseline
    python -m experiments.track_1_baseline --config config/train.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import structlog
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = structlog.get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "train.yaml"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic for binary classification."""
    from scipy.stats import ks_2samp

    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return 0.0
    stat, _ = ks_2samp(pos_probs, neg_probs)
    return float(stat)


def _gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Gini coefficient = 2 * ROC-AUC - 1."""
    return 2 * roc_auc_score(y_true, y_prob) - 1


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute all standard discrimination metrics."""
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "ks_statistic": _ks_statistic(y_true, y_prob),
        "gini": _gini(y_true, y_prob),
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _build_models(seed: int) -> dict[str, Any]:
    """Instantiate baseline models with default hyperparameters."""
    import lightgbm as lgb
    import xgboost as xgb

    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            solver="saga",
            random_state=seed,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=seed,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def main(config_path: Path | None = None) -> pd.DataFrame:
    """Run Track 1: baseline model comparison.

    Parameters
    ----------
    config_path : Path, optional
        Path to ``config/train.yaml``.

    Returns
    -------
    pd.DataFrame
        Summary table of model performance.
    """
    cfg_path = config_path or _DEFAULT_CONFIG
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    data_cfg = cfg["data"]
    mlflow_cfg = cfg.get("mlflow", {})

    # -- Load features -------------------------------------------------------
    features_path = _PROJECT_ROOT / data_cfg["features_path"]
    if not features_path.exists():
        logger.error("features_file_missing", path=str(features_path))
        raise FileNotFoundError(
            f"Features file not found: {features_path}. Run `make features` first."
        )

    df = pd.read_parquet(features_path)
    target_col = data_cfg["target_column"]
    id_col = data_cfg["id_column"]

    X = df.drop(columns=[target_col, id_col], errors="ignore")
    y = df[target_col].values

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=seed,
    )

    logger.info(
        "data_loaded",
        n_train=len(X_train),
        n_val=len(X_val),
        n_features=X_train.shape[1],
        pos_rate=float(y.mean()),
    )

    # -- MLflow setup --------------------------------------------------------
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns/"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "creditguard") + "/track_1_baseline")

    models = _build_models(seed)
    results: list[dict[str, Any]] = []

    for name, model in models.items():
        logger.info("training_model", model=name)

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)
            mlflow.log_param("seed", seed)
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_val", len(X_val))

            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]

            metrics = compute_metrics(y_val, y_prob)
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, round(value, 6))

            # Confusion matrix artifact
            y_pred = (y_prob >= 0.5).astype(int)
            cm = confusion_matrix(y_val, y_pred)
            cm_path = _PROJECT_ROOT / "mlflow" / f"cm_{name}.txt"
            cm_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(str(cm_path), cm, fmt="%d")
            mlflow.log_artifact(str(cm_path))

            row = {"model": name, **metrics}
            results.append(row)
            logger.info("model_complete", model=name, **metrics)

    # -- Summary table -------------------------------------------------------
    summary = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print("\n" + "=" * 70)
    print("TRACK 1 -- BASELINE MODEL COMPARISON")
    print("=" * 70)
    print(summary.to_string(index=False, float_format="%.4f"))
    print("=" * 70 + "\n")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Track 1: Baseline model comparison")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    args = parser.parse_args()
    main(config_path=args.config)
