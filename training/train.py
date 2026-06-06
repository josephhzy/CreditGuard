"""Main training pipeline for credit scoring models.

Config-driven pipeline that loads processed features, runs feature
selection, splits data, trains a model (optionally with HPO), evaluates
on held-out folds, logs everything to MLflow, and saves the final
model artifact.

Typical usage::

    python -m training.train --config config/train.yaml
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import structlog
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from training.cv_pipeline import get_cv_splits
from training.feature_selection import select_features
from training.hyperparameter_search import run_search

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    """Summary of a completed training run."""

    model_type: str
    n_features: int
    feature_names: list[str]
    cv_strategy: str
    n_folds: int
    fold_metrics: list[dict[str, float]]
    mean_roc_auc: float
    mean_pr_auc: float
    best_params: dict[str, Any]
    artifact_path: str
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _build_model(model_type: str, params: dict[str, Any]) -> Any:
    """Instantiate a scikit-learn compatible model.

    Parameters
    ----------
    model_type:
        One of ``"lightgbm"``, ``"xgboost"``, ``"rf"``, ``"logistic"``,
        ``"balanced_rf"``.
    params:
        Hyperparameter dict.

    Returns
    -------
    A fitted-ready estimator.
    """
    import lightgbm as lgb
    import xgboost as xgb
    from imblearn.ensemble import BalancedRandomForestClassifier

    clean = {k: v for k, v in params.items() if v is not None}

    builders: dict[str, Any] = {
        "lightgbm": lambda: lgb.LGBMClassifier(verbose=-1, n_jobs=-1, random_state=42, **clean),
        "xgboost": lambda: xgb.XGBClassifier(
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
            random_state=42,
            **clean,
        ),
        "rf": lambda: RandomForestClassifier(n_jobs=-1, random_state=42, **clean),
        "balanced_rf": lambda: BalancedRandomForestClassifier(n_jobs=-1, random_state=42, **clean),
        "logistic": lambda: LogisticRegression(random_state=42, max_iter=1000, **clean),
    }

    if model_type not in builders:
        raise ValueError(f"Unknown model type: '{model_type}'")
    return builders[model_type]()


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def train_pipeline(config_path: Path) -> TrainResult:
    """Execute the full training pipeline.

    1. Load features from parquet.
    2. Run feature selection (if configured).
    3. Split data via the configured CV strategy.
    4. Optionally run hyperparameter search.
    5. Train on each fold, collect metrics.
    6. Retrain on full data with best params.
    7. Log to MLflow, save model artifact.

    Parameters
    ----------
    config_path:
        Path to ``config/train.yaml``.

    Returns
    -------
    TrainResult
        Summary of the training run.
    """
    t_start = time.time()

    # ── Load config ───────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    project_root = config_path.resolve().parent.parent
    seed = cfg.get("seed", 42)
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    fs_cfg = cfg["feature_selection"]
    mlflow_cfg = cfg["mlflow"]

    # ── Load data ─────────────────────────────────────────
    features_path = project_root / data_cfg["features_path"]
    logger.info("loading_features", path=str(features_path))
    df = pd.read_parquet(features_path)

    target_col = data_cfg["target_column"]
    id_col = data_cfg["id_column"]

    non_feature_cols = [target_col, id_col]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    X = df[feature_cols]
    # Drop non-numeric columns (e.g. categorical mode strings) — trees need numerics
    numeric_mask = X.dtypes.apply(lambda dt: np.issubdtype(dt, np.number))
    non_numeric = numeric_mask[~numeric_mask].index.tolist()
    if non_numeric:
        logger.info("dropping_non_numeric_features", columns=non_numeric)
        X = X[numeric_mask[numeric_mask].index.tolist()]
        feature_cols = X.columns.tolist()
    y = df[target_col]

    logger.info(
        "data_loaded",
        n_rows=len(df),
        n_features=len(feature_cols),
        pos_rate=round(float(y.mean()), 4),
    )

    # ── Cross-validation splits ───────────────────────────
    cv_cfg = train_cfg["cv"]
    splits, _ = get_cv_splits(
        df,
        strategy=cv_cfg["strategy"],
        n_splits=cv_cfg["n_splits"],
        target_col=target_col,
        group_col=id_col,
    )

    # ── Feature selection ─────────────────────────────────
    # Selection runs once per fold on the fold's training rows so the OOF
    # metrics are honest (Boruta/RFE both consult labels, so running on the
    # full ``X, y`` would leak validation rows into the chosen feature set).
    # The final retrain on full data still selects on full data — the
    # production model is the production model — but the metrics that get
    # promoted are the per-fold ones.
    method = fs_cfg.get("method")
    do_feature_selection = bool(method and method != "none")
    if not do_feature_selection:
        logger.info("feature_selection_skipped")

    # ── Feature selection on full data (for HPO + final retrain only) ────
    # This selection IS allowed to see the full label distribution because
    # the resulting model is the production model — the only metrics we
    # promote as honest are the per-fold OOF metrics below, which use
    # fold-local selection. The full-data selection is ``X_full_selected``.
    fs_report_full: Any = None
    if do_feature_selection:
        logger.info("feature_selection_full_data", method=method)
        selected_features_full, fs_report_full = select_features(X, y, method=method)
        X_full_selected = X[selected_features_full]
        logger.info("features_after_full_selection", n_features=len(selected_features_full))
    else:
        selected_features_full = list(feature_cols)
        X_full_selected = X

    # ── Hyperparameter search (on first fold) ─────────────
    model_type = train_cfg["model"]
    hpo_cfg = train_cfg.get("hyperparameter_search", {})
    best_params: dict[str, Any] = {}

    if hpo_cfg.get("enabled", False):
        first_train_idx = splits[0][0]
        X_hpo = X_full_selected.iloc[first_train_idx]
        y_hpo = y.iloc[first_train_idx]

        logger.info("hpo_start", model=model_type, n_trials=hpo_cfg["n_trials"])
        best_params, hpo_report = run_search(
            X_hpo,
            y_hpo,
            model_type=model_type,
            n_trials=hpo_cfg["n_trials"],
            random_state=seed,
        )
        logger.info("hpo_done", best_score=round(hpo_report.best_score, 5))

    # ── MLflow setup ──────────────────────────────────────
    tracking_path = (project_root / mlflow_cfg["tracking_uri"]).resolve()
    tracking_uri = tracking_path.as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    # ── Cross-validation training ─────────────────────────
    fold_metrics: list[dict[str, float]] = []
    fold_selected_features: list[list[str]] = []
    oof_scores = np.zeros(len(y))
    oof_mask = np.zeros(len(y), dtype=bool)

    with mlflow.start_run(run_name=f"{model_type}_cv"):
        mlflow.log_params(
            {
                "model_type": model_type,
                "cv_strategy": cv_cfg["strategy"],
                "n_splits": cv_cfg["n_splits"],
                "feature_selection": fs_cfg.get("method", "none"),
                "n_features_full_selection": len(selected_features_full),
                "feature_selection_inside_cv": do_feature_selection,
                "seed": seed,
            }
        )
        if best_params:
            mlflow.log_params({f"hp_{k}": v for k, v in best_params.items()})

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Per-fold feature selection on the fold's training rows only —
            # the chosen features may not match ``selected_features_full``.
            # That is the point: it stops Boruta/RFE from seeing val labels.
            if do_feature_selection:
                fold_selected, _ = select_features(X.iloc[train_idx], y_train, method=method)
            else:
                fold_selected = list(feature_cols)
            fold_selected_features.append(fold_selected)

            X_train = X.iloc[train_idx][fold_selected]
            X_val = X.iloc[val_idx][fold_selected]

            # Scale for logistic regression
            scaler = None
            if model_type == "logistic":
                scaler = StandardScaler()
                X_train = pd.DataFrame(
                    scaler.fit_transform(X_train), columns=fold_selected, index=X_train.index
                )
                X_val = pd.DataFrame(
                    scaler.transform(X_val), columns=fold_selected, index=X_val.index
                )

            model = _build_model(model_type, best_params)
            model.fit(X_train, y_train)

            y_val_proba = model.predict_proba(X_val)[:, 1]

            fold_roc = roc_auc_score(y_val, y_val_proba)
            fold_pr = average_precision_score(y_val, y_val_proba)

            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "roc_auc": fold_roc,
                    "pr_auc": fold_pr,
                    "n_features": len(fold_selected),
                }
            )
            oof_scores[val_idx] = y_val_proba
            oof_mask[val_idx] = True

            logger.info(
                "fold_complete",
                fold=fold_idx,
                roc_auc=round(fold_roc, 5),
                pr_auc=round(fold_pr, 5),
                n_features=len(fold_selected),
            )

        # ── Aggregate metrics ─────────────────────────────
        mean_roc = float(np.mean([m["roc_auc"] for m in fold_metrics]))
        mean_pr = float(np.mean([m["pr_auc"] for m in fold_metrics]))
        std_roc = float(np.std([m["roc_auc"] for m in fold_metrics]))
        std_pr = float(np.std([m["pr_auc"] for m in fold_metrics]))

        # OOF metrics (covers all data that appeared in at least one val fold)
        oof_roc = roc_auc_score(y[oof_mask], oof_scores[oof_mask])
        oof_pr = average_precision_score(y[oof_mask], oof_scores[oof_mask])

        mlflow.log_metrics(
            {
                "mean_roc_auc": mean_roc,
                "mean_pr_auc": mean_pr,
                "std_roc_auc": std_roc,
                "std_pr_auc": std_pr,
                "oof_roc_auc": oof_roc,
                "oof_pr_auc": oof_pr,
            }
        )

        logger.info(
            "cv_summary",
            mean_roc_auc=round(mean_roc, 5),
            mean_pr_auc=round(mean_pr, 5),
            oof_roc_auc=round(oof_roc, 5),
            oof_pr_auc=round(oof_pr, 5),
        )

        # ── Retrain on full data ──────────────────────────
        logger.info(
            "retraining_on_full_data",
            n_features=len(selected_features_full),
        )
        scaler_final = None
        X_full = X_full_selected.copy()
        if model_type == "logistic":
            scaler_final = StandardScaler()
            X_full = pd.DataFrame(
                scaler_final.fit_transform(X_full),
                columns=selected_features_full,
                index=X_full.index,
            )

        final_model = _build_model(model_type, best_params)
        final_model.fit(X_full, y)

        # ── Save artifact ─────────────────────────────────
        artifact_dir = project_root / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / f"{model_type}_model.pkl"

        # Fit Platt scaling on the OOF scores so the bundle ships with a
        # calibrator. The README documents ``risk_score`` as a calibrated PD;
        # without this the served field would be a raw classifier output.
        calibrator = None
        oof_y = y[oof_mask].to_numpy()
        oof_s = oof_scores[oof_mask].reshape(-1, 1)
        if len(oof_y) > 0 and 0 < oof_y.mean() < 1:
            from sklearn.linear_model import LogisticRegression

            calibrator = LogisticRegression(C=1e9, solver="lbfgs", max_iter=200)
            calibrator.fit(oof_s, oof_y)

        artifact = {
            "model": final_model,
            "scaler": scaler_final,
            "calibrator": calibrator,
            "calibrator_method": "platt_logistic_on_oof" if calibrator is not None else None,
            "feature_names": selected_features_full,
            "fold_selected_features": fold_selected_features,
            "params": best_params,
            "model_type": model_type,
        }
        with open(model_path, "wb") as f:
            pickle.dump(artifact, f)

        mlflow.log_artifact(str(model_path))
        mlflow.log_dict(
            {
                "features_full_selection": selected_features_full,
                "fold_selected_features": fold_selected_features,
                "params": best_params,
            },
            "model_metadata.json",
        )

        logger.info("model_saved", path=str(model_path))

    elapsed = time.time() - t_start

    result = TrainResult(
        model_type=model_type,
        n_features=len(selected_features_full),
        feature_names=selected_features_full,
        cv_strategy=cv_cfg["strategy"],
        n_folds=len(splits),
        fold_metrics=fold_metrics,
        mean_roc_auc=mean_roc,
        mean_pr_auc=mean_pr,
        best_params=best_params,
        artifact_path=str(model_path),
        elapsed_seconds=round(elapsed, 1),
    )

    logger.info(
        "training_complete",
        model=model_type,
        mean_roc_auc=round(mean_roc, 5),
        mean_pr_auc=round(mean_pr, 5),
        elapsed_s=result.elapsed_seconds,
    )
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the CreditGuard training pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/train.yaml"),
        help="Path to training config YAML",
    )
    args = parser.parse_args()

    if not args.config.exists():
        logger.error("config_not_found", path=str(args.config))
        raise SystemExit(1)

    result = train_pipeline(args.config)

    print(f"\n{'=' * 60}")
    print(f"Training Complete: {result.model_type}")
    print(f"{'=' * 60}")
    print(f"  Features:     {result.n_features}")
    print(f"  CV Strategy:  {result.cv_strategy} ({result.n_folds} folds)")
    print(f"  Mean ROC-AUC: {result.mean_roc_auc:.5f}")
    print(f"  Mean PR-AUC:  {result.mean_pr_auc:.5f}")
    print(f"  Artifact:     {result.artifact_path}")
    print(f"  Elapsed:      {result.elapsed_seconds}s")
