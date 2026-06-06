"""Deliberate experiment tracks for CreditGuard model development.

Five systematic experiment tracks, each logged to MLflow:

- Track 1: Baseline models (LR, RF, XGBoost, LightGBM)
- Track 2: Imbalance treatment (class_weight, SMOTE, ADASYN, etc.)
- Track 3: Feature selection (none, Boruta, RFE, VIF, hybrid)
- Track 4: Validation design (random stratified, grouped CV, OOT)
- Track 5: Calibration (raw, Platt scaling, isotonic regression)

Typical usage::

    python -m training.experiments --track 1 --config config/train.yaml
    python -m training.experiments --track all
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import mlflow
import numpy as np
import pandas as pd
import structlog
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from training.cv_pipeline import get_cv_splits
from training.feature_selection import select_features

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ExperimentResult:
    """Result of a single experiment within a track."""

    track: int
    experiment_name: str
    variant: str
    roc_auc: float
    pr_auc: float
    extra_metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


@dataclass
class TrackReport:
    """Summary of all experiments in a track."""

    track: int
    track_name: str
    results: list[ExperimentResult]
    best_variant: str
    best_pr_auc: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_data(cfg: dict, project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load features, return (df, X, y)."""
    features_path = project_root / cfg["data"]["features_path"]
    df = pd.read_parquet(features_path)
    target_col = cfg["data"]["target_column"]
    id_col = cfg["data"]["id_column"]
    X = df.drop(columns=[target_col, id_col], errors="ignore")
    y = df[target_col]
    return df, X, y


def _setup_mlflow(cfg: dict, project_root: Path, experiment_name: str) -> None:
    """Configure MLflow tracking."""
    tracking_uri = str(project_root / cfg["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def _cv_evaluate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Evaluate a model via stratified CV, returning ROC-AUC, PR-AUC, and OOF scores."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_scores = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    roc = roc_auc_score(y, y_scores)
    pr = average_precision_score(y, y_scores)
    return roc, pr, y_scores


# ---------------------------------------------------------------------------
# Track 1: Baseline models
# ---------------------------------------------------------------------------


def run_track_1_baseline(cfg: dict, project_root: Path) -> TrackReport:
    """Compare baseline models with default hyperparameters.

    Models: LogisticRegression, RandomForest, XGBoost, LightGBM.
    """
    import lightgbm as lgb
    import xgboost as xgb

    _setup_mlflow(cfg, project_root, "track_1_baseline")
    _, X, y = _load_data(cfg, project_root)

    models = {
        "logistic": LogisticRegression(max_iter=1000, random_state=42),
        "rf": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        "xgboost": xgb.XGBClassifier(
            n_estimators=200,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
            random_state=42,
        ),
        "lightgbm": lgb.LGBMClassifier(n_estimators=200, verbose=-1, n_jobs=-1, random_state=42),
    }

    results: list[ExperimentResult] = []

    for name, model in models.items():
        logger.info("track_1_running", model=name)
        t0 = time.time()

        with mlflow.start_run(run_name=f"baseline_{name}"):
            roc, pr, _ = _cv_evaluate(model, X, y)
            elapsed = time.time() - t0

            mlflow.log_params({"model": name, "track": 1})
            mlflow.log_metrics({"roc_auc": roc, "pr_auc": pr})

            result = ExperimentResult(
                track=1,
                experiment_name="baseline",
                variant=name,
                roc_auc=roc,
                pr_auc=pr,
                elapsed_seconds=round(elapsed, 1),
            )
            results.append(result)
            logger.info("track_1_result", model=name, roc_auc=round(roc, 5), pr_auc=round(pr, 5))

    best = max(results, key=lambda r: r.pr_auc)
    return TrackReport(
        track=1,
        track_name="baseline",
        results=results,
        best_variant=best.variant,
        best_pr_auc=best.pr_auc,
    )


# ---------------------------------------------------------------------------
# Track 2: Imbalance treatment
# ---------------------------------------------------------------------------


def run_track_2_imbalance(cfg: dict, project_root: Path) -> TrackReport:
    """Compare imbalance handling strategies.

    Variants: class_weight, SMOTE, ADASYN, EasyEnsemble, BalancedRF, RUSBoost.
    """
    import lightgbm as lgb
    from imblearn.ensemble import (
        BalancedRandomForestClassifier,
        EasyEnsembleClassifier,
        RUSBoostClassifier,
    )
    from imblearn.over_sampling import ADASYN, SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    _setup_mlflow(cfg, project_root, "track_2_imbalance")
    _, X, y = _load_data(cfg, project_root)

    base_lgb = lgb.LGBMClassifier(n_estimators=200, verbose=-1, n_jobs=-1, random_state=42)

    variants: dict[str, Any] = {
        "no_treatment": base_lgb,
        "class_weight": lgb.LGBMClassifier(
            n_estimators=200,
            verbose=-1,
            n_jobs=-1,
            random_state=42,
            is_unbalance=True,
        ),
        "smote": ImbPipeline(
            [
                ("smote", SMOTE(random_state=42)),
                (
                    "clf",
                    lgb.LGBMClassifier(n_estimators=200, verbose=-1, n_jobs=-1, random_state=42),
                ),
            ]
        ),
        "adasyn": ImbPipeline(
            [
                ("adasyn", ADASYN(random_state=42)),
                (
                    "clf",
                    lgb.LGBMClassifier(n_estimators=200, verbose=-1, n_jobs=-1, random_state=42),
                ),
            ]
        ),
        "easy_ensemble": EasyEnsembleClassifier(n_estimators=20, random_state=42),
        "balanced_rf": BalancedRandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        "rusboost": RUSBoostClassifier(n_estimators=200, random_state=42),
    }

    results: list[ExperimentResult] = []

    for name, model in variants.items():
        logger.info("track_2_running", variant=name)
        t0 = time.time()

        with mlflow.start_run(run_name=f"imbalance_{name}"):
            roc, pr, _ = _cv_evaluate(model, X, y)
            elapsed = time.time() - t0

            mlflow.log_params({"variant": name, "track": 2})
            mlflow.log_metrics({"roc_auc": roc, "pr_auc": pr})

            result = ExperimentResult(
                track=2,
                experiment_name="imbalance",
                variant=name,
                roc_auc=roc,
                pr_auc=pr,
                elapsed_seconds=round(elapsed, 1),
            )
            results.append(result)
            logger.info("track_2_result", variant=name, roc_auc=round(roc, 5), pr_auc=round(pr, 5))

    best = max(results, key=lambda r: r.pr_auc)
    return TrackReport(
        track=2,
        track_name="imbalance",
        results=results,
        best_variant=best.variant,
        best_pr_auc=best.pr_auc,
    )


# ---------------------------------------------------------------------------
# Track 3: Feature selection
# ---------------------------------------------------------------------------


def run_track_3_feature_selection(cfg: dict, project_root: Path) -> TrackReport:
    """Compare feature selection strategies.

    Variants: none, Boruta, RFE, VIF, hybrid.
    """
    import lightgbm as lgb

    _setup_mlflow(cfg, project_root, "track_3_feature_selection")
    _, X, y = _load_data(cfg, project_root)

    methods: list[Literal["none", "boruta", "rfe", "vif", "hybrid"]] = [
        "none",
        "boruta",
        "rfe",
        "vif",
        "hybrid",
    ]
    results: list[ExperimentResult] = []

    for method in methods:
        logger.info("track_3_running", method=method)
        t0 = time.time()

        with mlflow.start_run(run_name=f"fs_{method}"):
            if method == "none":
                X_sel = X
            else:
                selected, fs_report = select_features(X, y, method=method)
                X_sel = X[selected]

            model = lgb.LGBMClassifier(n_estimators=200, verbose=-1, n_jobs=-1, random_state=42)
            roc, pr, _ = _cv_evaluate(model, X_sel, y)
            elapsed = time.time() - t0

            mlflow.log_params({"method": method, "n_features": X_sel.shape[1], "track": 3})
            mlflow.log_metrics({"roc_auc": roc, "pr_auc": pr})

            result = ExperimentResult(
                track=3,
                experiment_name="feature_selection",
                variant=method,
                roc_auc=roc,
                pr_auc=pr,
                extra_metrics={"n_features": X_sel.shape[1]},
                elapsed_seconds=round(elapsed, 1),
            )
            results.append(result)
            logger.info(
                "track_3_result",
                method=method,
                n_features=X_sel.shape[1],
                roc_auc=round(roc, 5),
                pr_auc=round(pr, 5),
            )

    best = max(results, key=lambda r: r.pr_auc)
    return TrackReport(
        track=3,
        track_name="feature_selection",
        results=results,
        best_variant=best.variant,
        best_pr_auc=best.pr_auc,
    )


# ---------------------------------------------------------------------------
# Track 4: Validation design
# ---------------------------------------------------------------------------


def run_track_4_validation(cfg: dict, project_root: Path) -> TrackReport:
    """Compare validation strategies.

    Variants: random stratified, grouped CV (prevents applicant leakage),
    out-of-time (OOT) if date column is available.
    """
    import lightgbm as lgb

    _setup_mlflow(cfg, project_root, "track_4_validation")
    df, X, y = _load_data(cfg, project_root)
    target_col = cfg["data"]["target_column"]
    id_col = cfg["data"]["id_column"]

    strategies = ["stratified", "stratified_group"]

    # Check if a date column exists for OOT
    date_candidates = [c for c in df.columns if "date" in c.lower() or "days" in c.lower()]
    if date_candidates:
        strategies.append("temporal")

    results: list[ExperimentResult] = []

    for strategy in strategies:
        logger.info("track_4_running", strategy=strategy)
        t0 = time.time()

        with mlflow.start_run(run_name=f"val_{strategy}"):
            date_col = date_candidates[0] if strategy == "temporal" and date_candidates else None
            splits, _ = get_cv_splits(
                df,
                strategy=strategy,
                n_splits=5,
                target_col=target_col,
                group_col=id_col,
                date_col=date_col,
            )

            fold_rocs, fold_prs = [], []
            for train_idx, val_idx in splits:
                X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

                m = lgb.LGBMClassifier(n_estimators=200, verbose=-1, n_jobs=-1, random_state=42)
                m.fit(X_tr, y_tr)
                y_prob = m.predict_proba(X_va)[:, 1]
                fold_rocs.append(roc_auc_score(y_va, y_prob))
                fold_prs.append(average_precision_score(y_va, y_prob))

            roc = float(np.mean(fold_rocs))
            pr = float(np.mean(fold_prs))
            elapsed = time.time() - t0

            mlflow.log_params({"strategy": strategy, "n_folds": len(splits), "track": 4})
            mlflow.log_metrics(
                {
                    "roc_auc": roc,
                    "pr_auc": pr,
                    "std_roc_auc": float(np.std(fold_rocs)),
                    "std_pr_auc": float(np.std(fold_prs)),
                }
            )

            result = ExperimentResult(
                track=4,
                experiment_name="validation",
                variant=strategy,
                roc_auc=roc,
                pr_auc=pr,
                extra_metrics={"n_folds": len(splits)},
                elapsed_seconds=round(elapsed, 1),
            )
            results.append(result)
            logger.info(
                "track_4_result", strategy=strategy, roc_auc=round(roc, 5), pr_auc=round(pr, 5)
            )

    best = max(results, key=lambda r: r.pr_auc)
    return TrackReport(
        track=4,
        track_name="validation",
        results=results,
        best_variant=best.variant,
        best_pr_auc=best.pr_auc,
    )


# ---------------------------------------------------------------------------
# Track 5: Calibration
# ---------------------------------------------------------------------------


def run_track_5_calibration(cfg: dict, project_root: Path) -> TrackReport:
    """Compare calibration methods.

    Variants: raw (uncalibrated), Platt scaling (sigmoid), isotonic regression.
    """
    import lightgbm as lgb

    _setup_mlflow(cfg, project_root, "track_5_calibration")
    _, X, y = _load_data(cfg, project_root)

    base_model = lgb.LGBMClassifier(n_estimators=200, verbose=-1, n_jobs=-1, random_state=42)

    variants: dict[str, Any] = {
        "raw": base_model,
        "platt": CalibratedClassifierCV(base_model, method="sigmoid", cv=3),
        "isotonic": CalibratedClassifierCV(base_model, method="isotonic", cv=3),
    }

    results: list[ExperimentResult] = []

    for name, model in variants.items():
        logger.info("track_5_running", variant=name)
        t0 = time.time()

        with mlflow.start_run(run_name=f"cal_{name}"):
            roc, pr, y_scores = _cv_evaluate(model, X, y)
            brier = brier_score_loss(y, y_scores)
            elapsed = time.time() - t0

            mlflow.log_params({"variant": name, "track": 5})
            mlflow.log_metrics({"roc_auc": roc, "pr_auc": pr, "brier_score": brier})

            result = ExperimentResult(
                track=5,
                experiment_name="calibration",
                variant=name,
                roc_auc=roc,
                pr_auc=pr,
                extra_metrics={"brier_score": brier},
                elapsed_seconds=round(elapsed, 1),
            )
            results.append(result)
            logger.info(
                "track_5_result",
                variant=name,
                roc_auc=round(roc, 5),
                pr_auc=round(pr, 5),
                brier=round(brier, 5),
            )

    best = max(results, key=lambda r: r.pr_auc)
    return TrackReport(
        track=5,
        track_name="calibration",
        results=results,
        best_variant=best.variant,
        best_pr_auc=best.pr_auc,
    )


# ---------------------------------------------------------------------------
# Track dispatcher
# ---------------------------------------------------------------------------

_TRACK_RUNNERS = {
    1: run_track_1_baseline,
    2: run_track_2_imbalance,
    3: run_track_3_feature_selection,
    4: run_track_4_validation,
    5: run_track_5_calibration,
}


def run_track(track: int, cfg: dict, project_root: Path) -> TrackReport:
    """Run a single experiment track.

    Parameters
    ----------
    track:
        Track number (1-5).
    cfg:
        Loaded config dict.
    project_root:
        Project root path.

    Returns
    -------
    TrackReport
    """
    if track not in _TRACK_RUNNERS:
        raise ValueError(f"Unknown track: {track}. Valid: {list(_TRACK_RUNNERS.keys())}")
    return _TRACK_RUNNERS[track](cfg, project_root)


def run_all_tracks(cfg: dict, project_root: Path) -> list[TrackReport]:
    """Run all five experiment tracks sequentially.

    Parameters
    ----------
    cfg:
        Loaded config dict.
    project_root:
        Project root path.

    Returns
    -------
    list[TrackReport]
    """
    reports = []
    for track_num in sorted(_TRACK_RUNNERS.keys()):
        logger.info("starting_track", track=track_num)
        report = run_track(track_num, cfg, project_root)
        reports.append(report)
        logger.info(
            "track_complete",
            track=track_num,
            best_variant=report.best_variant,
            best_pr_auc=round(report.best_pr_auc, 5),
        )
    return reports


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CreditGuard experiment tracks")
    parser.add_argument("--track", type=str, default="all", help="Track number (1-5) or 'all'")
    parser.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_root = args.config.resolve().parent.parent

    if args.track == "all":
        reports = run_all_tracks(cfg, project_root)
        print(f"\n{'=' * 60}")
        print("Experiment Summary")
        print(f"{'=' * 60}")
        for tr in reports:
            print(
                f"  Track {tr.track} ({tr.track_name}): best={tr.best_variant} PR-AUC={tr.best_pr_auc:.5f}"
            )
    else:
        track_num = int(args.track)
        report = run_track(track_num, cfg, project_root)
        print(f"\nTrack {report.track} ({report.track_name})")
        print(f"  Best variant: {report.best_variant} (PR-AUC={report.best_pr_auc:.5f})")
        for er in report.results:
            print(
                f"    {er.variant}: ROC={er.roc_auc:.5f} PR={er.pr_auc:.5f} ({er.elapsed_seconds}s)"
            )
