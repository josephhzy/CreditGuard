"""Track 2 -- Imbalance treatment comparison.

Compares different strategies for handling class imbalance:
  - class_weight="balanced" (native LightGBM / RF)
  - SMOTE oversampling
  - ADASYN adaptive oversampling
  - EasyEnsembleClassifier
  - BalancedRandomForestClassifier
  - RUSBoostClassifier

All strategies use the same base model and evaluation split.

Usage::

    python -m experiments.track_2_imbalance
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import pandas as pd
import structlog
import yaml
from imblearn.ensemble import (
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier,
)
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split

from experiments.track_1_baseline import compute_metrics

logger = structlog.get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "train.yaml"


# ---------------------------------------------------------------------------
# Strategy builders
# ---------------------------------------------------------------------------


def _build_strategies(seed: int) -> dict[str, Any]:
    """Return a dict of {strategy_name: fitted-estimator-or-pipeline}."""
    base_params: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }

    strategies: dict[str, Any] = {}

    # 1. Balanced class weights (native)
    strategies["balanced_weight"] = lgb.LGBMClassifier(
        **base_params,
        is_unbalance=True,
    )

    # 2. SMOTE + LightGBM
    strategies["SMOTE"] = ImbPipeline(
        [
            ("smote", SMOTE(random_state=seed)),
            ("lgbm", lgb.LGBMClassifier(**base_params)),
        ]
    )

    # 3. ADASYN + LightGBM
    strategies["ADASYN"] = ImbPipeline(
        [
            ("adasyn", ADASYN(random_state=seed)),
            ("lgbm", lgb.LGBMClassifier(**base_params)),
        ]
    )

    # 4. EasyEnsembleClassifier
    strategies["EasyEnsemble"] = EasyEnsembleClassifier(
        n_estimators=20,
        random_state=seed,
        n_jobs=-1,
    )

    # 5. BalancedRandomForest
    strategies["BalancedRF"] = BalancedRandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=seed,
        n_jobs=-1,
    )

    # 6. RUSBoostClassifier
    strategies["RUSBoost"] = RUSBoostClassifier(
        n_estimators=200,
        random_state=seed,
    )

    return strategies


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def main(config_path: Path | None = None) -> pd.DataFrame:
    """Run Track 2: imbalance treatment comparison.

    Parameters
    ----------
    config_path : Path, optional
        Path to ``config/train.yaml``.

    Returns
    -------
    pd.DataFrame
        Summary table of strategy performance.
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
        raise FileNotFoundError(f"Features file not found: {features_path}")

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
        pos_rate_train=float(y_train.mean()),
    )

    # -- MLflow setup --------------------------------------------------------
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns/"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "creditguard") + "/track_2_imbalance")

    strategies = _build_strategies(seed)
    results: list[dict[str, Any]] = []

    for name, estimator in strategies.items():
        logger.info("training_strategy", strategy=name)

        with mlflow.start_run(run_name=name):
            mlflow.log_param("strategy", name)
            mlflow.log_param("seed", seed)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("pos_rate_train", round(float(y_train.mean()), 4))

            try:
                estimator.fit(X_train, y_train)
                y_prob = estimator.predict_proba(X_val)[:, 1]
                metrics = compute_metrics(y_val, y_prob)

                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, round(value, 6))

                row = {"strategy": name, **metrics}
                results.append(row)
                logger.info("strategy_complete", strategy=name, **metrics)
            except Exception as exc:
                logger.error("strategy_failed", strategy=name, error=str(exc))
                mlflow.log_param("error", str(exc))
                results.append(
                    {
                        "strategy": name,
                        "roc_auc": 0.0,
                        "pr_auc": 0.0,
                        "ks_statistic": 0.0,
                        "gini": 0.0,
                    }
                )

    # -- Summary table -------------------------------------------------------
    summary = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    print("\n" + "=" * 70)
    print("TRACK 2 -- IMBALANCE TREATMENT COMPARISON")
    print("=" * 70)
    print(summary.to_string(index=False, float_format="%.4f"))
    print("=" * 70 + "\n")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Track 2: Imbalance treatment comparison")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    main(config_path=args.config)
