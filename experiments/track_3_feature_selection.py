"""Track 3 -- Feature selection comparison.

Compares different feature selection strategies:
  - No selection (all features)
  - Boruta
  - Recursive Feature Elimination (RFE)
  - Variance Inflation Factor (VIF) filtering
  - Hybrid (Boruta + VIF)

All variants use the same LightGBM model and evaluation split.

Usage::

    python -m experiments.track_3_feature_selection
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
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

from experiments.track_1_baseline import compute_metrics

logger = structlog.get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "train.yaml"


# ---------------------------------------------------------------------------
# Feature selection methods
# ---------------------------------------------------------------------------


def _select_boruta(X: pd.DataFrame, y: np.ndarray, cfg: dict, seed: int) -> list[str]:
    """Run Boruta feature selection and return selected feature names."""
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestClassifier

    boruta_cfg = cfg.get("feature_selection", {}).get("boruta", {})
    max_iter = boruta_cfg.get("max_iter", 200)
    alpha = boruta_cfg.get("alpha", 0.05)

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1)
    selector = BorutaPy(
        rf, n_estimators="auto", max_iter=max_iter, alpha=alpha, random_state=seed, verbose=0
    )
    selector.fit(X.values, y)

    selected = X.columns[selector.support_].tolist()
    logger.info("boruta_complete", n_selected=len(selected), n_total=X.shape[1])
    return selected


def _select_rfe(X: pd.DataFrame, y: np.ndarray, cfg: dict, seed: int) -> list[str]:
    """Run RFE and return selected feature names."""
    rfe_cfg = cfg.get("feature_selection", {}).get("rfe", {})
    n_features = rfe_cfg.get("n_features_to_select", 50)
    step = rfe_cfg.get("step", 5)

    estimator = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    )
    rfe = RFE(estimator, n_features_to_select=min(n_features, X.shape[1]), step=step)
    rfe.fit(X, y)

    selected = X.columns[rfe.support_].tolist()
    logger.info("rfe_complete", n_selected=len(selected))
    return selected


def _select_vif(X: pd.DataFrame, cfg: dict) -> list[str]:
    """Remove features with VIF above the threshold iteratively."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_cfg = cfg.get("feature_selection", {}).get("vif", {})
    threshold = vif_cfg.get("threshold", 10.0)

    # Work on numeric columns only; fill NaN for VIF computation
    X_vif = X.select_dtypes(include=[np.number]).fillna(0).copy()

    dropped: list[str] = []
    while True:
        if X_vif.shape[1] < 2:
            break
        vifs = pd.Series(
            [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
            index=X_vif.columns,
        )
        max_vif = vifs.max()
        if max_vif <= threshold:
            break
        worst = vifs.idxmax()
        dropped.append(worst)
        X_vif = X_vif.drop(columns=[worst])

    selected = X_vif.columns.tolist()
    logger.info("vif_complete", n_selected=len(selected), n_dropped=len(dropped))
    return selected


def _select_hybrid(X: pd.DataFrame, y: np.ndarray, cfg: dict, seed: int) -> list[str]:
    """Hybrid: Boruta first, then VIF filtering on the Boruta-selected set."""
    boruta_features = _select_boruta(X, y, cfg, seed)
    if not boruta_features:
        return boruta_features
    vif_features = _select_vif(X[boruta_features], cfg)
    logger.info("hybrid_complete", n_boruta=len(boruta_features), n_final=len(vif_features))
    return vif_features


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def main(config_path: Path | None = None) -> pd.DataFrame:
    """Run Track 3: feature selection comparison.

    Parameters
    ----------
    config_path : Path, optional
        Path to ``config/train.yaml``.

    Returns
    -------
    pd.DataFrame
        Summary table of selection strategy performance.
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

    logger.info("data_loaded", n_features=X.shape[1])

    # -- Feature selection strategies ----------------------------------------
    selection_methods: dict[str, list[str] | None] = {
        "all_features": None,  # use all
    }

    logger.info("running_boruta")
    try:
        selection_methods["boruta"] = _select_boruta(X_train, y_train, cfg, seed)
    except Exception as exc:
        logger.warning("boruta_failed", error=str(exc))

    logger.info("running_rfe")
    try:
        selection_methods["rfe"] = _select_rfe(X_train, y_train, cfg, seed)
    except Exception as exc:
        logger.warning("rfe_failed", error=str(exc))

    logger.info("running_vif")
    try:
        selection_methods["vif"] = _select_vif(X_train, cfg)
    except Exception as exc:
        logger.warning("vif_failed", error=str(exc))

    logger.info("running_hybrid")
    try:
        selection_methods["hybrid"] = _select_hybrid(X_train, y_train, cfg, seed)
    except Exception as exc:
        logger.warning("hybrid_failed", error=str(exc))

    # -- MLflow setup --------------------------------------------------------
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns/"))
    mlflow.set_experiment(
        mlflow_cfg.get("experiment_name", "creditguard") + "/track_3_feature_selection"
    )

    base_params: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }

    results: list[dict[str, Any]] = []

    for name, features in selection_methods.items():
        X_tr = (
            X_train if features is None else X_train[[f for f in features if f in X_train.columns]]
        )
        X_va = X_val if features is None else X_val[[f for f in features if f in X_val.columns]]
        n_feat = X_tr.shape[1]

        logger.info("training_variant", variant=name, n_features=n_feat)

        with mlflow.start_run(run_name=name):
            mlflow.log_param("selection_method", name)
            mlflow.log_param("n_features_selected", n_feat)
            mlflow.log_param("seed", seed)
            if features is not None:
                # Log selected feature list as artifact
                feat_path = _PROJECT_ROOT / "mlflow" / f"features_{name}.txt"
                feat_path.parent.mkdir(parents=True, exist_ok=True)
                feat_path.write_text("\n".join(features))
                mlflow.log_artifact(str(feat_path))

            model = lgb.LGBMClassifier(**base_params)
            model.fit(X_tr, y_train)
            y_prob = model.predict_proba(X_va)[:, 1]

            metrics = compute_metrics(y_val, y_prob)
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, round(value, 6))

            row = {"method": name, "n_features": n_feat, **metrics}
            results.append(row)
            logger.info("variant_complete", variant=name, **metrics)

    # -- Summary table -------------------------------------------------------
    summary = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    print("\n" + "=" * 70)
    print("TRACK 3 -- FEATURE SELECTION COMPARISON")
    print("=" * 70)
    print(summary.to_string(index=False, float_format="%.4f"))
    print("=" * 70 + "\n")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Track 3: Feature selection comparison")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    main(config_path=args.config)
