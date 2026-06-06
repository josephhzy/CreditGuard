"""Optuna-based hyperparameter optimization for credit scoring models.

Defines search spaces for LightGBM, XGBoost, RandomForest, and
LogisticRegression, with PR-AUC as the default optimization metric.

Typical usage::

    from training.hyperparameter_search import run_search

    best_params, study = run_search(
        X_train, y_train, model_type="lightgbm", n_trials=50,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
import structlog
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# Lazy imports for heavy dependencies
_optuna = None
_lgb = None
_xgb = None


def _import_optuna():  # type: ignore[return]
    global _optuna
    if _optuna is None:
        import optuna

        _optuna = optuna
    return _optuna


def _import_lgb():  # type: ignore[return]
    global _lgb
    if _lgb is None:
        import lightgbm as lgb

        _lgb = lgb
    return _lgb


def _import_xgb():  # type: ignore[return]
    global _xgb
    if _xgb is None:
        import xgboost as xgb

        _xgb = xgb
    return _xgb


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SearchReport:
    """Results of a hyperparameter search."""

    model_type: str
    n_trials: int
    best_score: float
    best_params: dict[str, Any]
    metric: str
    trial_history: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Search space definitions
# ---------------------------------------------------------------------------

ModelType = Literal["lightgbm", "xgboost", "rf", "logistic"]


def define_search_space(trial: Any, model_type: ModelType) -> dict[str, Any]:
    """Define the hyperparameter search space for a given model type.

    Parameters
    ----------
    trial:
        An Optuna trial object.
    model_type:
        One of ``"lightgbm"``, ``"xgboost"``, ``"rf"``, ``"logistic"``.

    Returns
    -------
    dict[str, Any]
        Hyperparameter dictionary sampled from the search space.
    """
    if model_type == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 15.0),
        }

    if model_type == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 15.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }

    if model_type == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7]),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample"]
            ),
        }

    if model_type == "logistic":
        return {
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
            "solver": "saga",  # supports all penalties
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0)
            if trial.params.get("penalty") == "elasticnet"
            else None,
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            "max_iter": 1000,
        }

    raise ValueError(f"Unknown model type: '{model_type}'")


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _build_model(model_type: ModelType, params: dict[str, Any]) -> Any:
    """Instantiate a model from type and hyperparameters.

    Parameters
    ----------
    model_type:
        Model family name.
    params:
        Hyperparameter dictionary.

    Returns
    -------
    A scikit-learn compatible estimator.
    """
    clean_params = {k: v for k, v in params.items() if v is not None}

    if model_type == "lightgbm":
        lgb = _import_lgb()
        return lgb.LGBMClassifier(verbose=-1, n_jobs=-1, random_state=42, **clean_params)

    if model_type == "xgboost":
        xgb = _import_xgb()
        return xgb.XGBClassifier(
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
            random_state=42,
            **clean_params,
        )

    if model_type == "rf":
        return RandomForestClassifier(n_jobs=-1, random_state=42, **clean_params)

    if model_type == "logistic":
        return LogisticRegression(random_state=42, **clean_params)

    raise ValueError(f"Unknown model type: '{model_type}'")


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------


def _make_objective(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: ModelType,
    n_cv_splits: int = 3,
    random_state: int = 42,
):
    """Create an Optuna objective function.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Binary target.
    model_type:
        Model family.
    n_cv_splits:
        Number of stratified CV folds for evaluation.
    random_state:
        Random seed.

    Returns
    -------
    callable
        An objective function compatible with ``optuna.Study.optimize``.
    """
    cv = StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=random_state)

    def objective(trial):  # type: ignore[no-untyped-def]
        params = define_search_space(trial, model_type)
        model = _build_model(model_type, params)

        try:
            y_scores = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
            score = average_precision_score(y, y_scores)
        except (ValueError, TypeError) as exc:
            # Model convergence or parameter incompatibility — prune trial
            logger.warning(
                "trial_pruned", trial=trial.number, error=str(exc), error_type=type(exc).__name__
            )
            raise _import_optuna().TrialPruned(str(exc)) from exc
        except Exception as exc:
            # Unexpected errors (data issues, memory) — re-raise to surface bugs
            logger.error(
                "trial_error", trial=trial.number, error=str(exc), error_type=type(exc).__name__
            )
            raise

        return score

    return objective


# ---------------------------------------------------------------------------
# Main search function
# ---------------------------------------------------------------------------


def run_search(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: ModelType = "lightgbm",
    n_trials: int = 50,
    metric: str = "pr_auc",
    n_cv_splits: int = 3,
    random_state: int = 42,
    timeout: int | None = None,
) -> tuple[dict[str, Any], SearchReport]:
    """Run Optuna hyperparameter search.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Binary target.
    model_type:
        Model family to optimize.
    n_trials:
        Number of Optuna trials.
    metric:
        Metric name (for logging; always optimizes PR-AUC internally).
    n_cv_splits:
        CV folds for inner evaluation.
    random_state:
        Random seed.
    timeout:
        Optional time limit in seconds.

    Returns
    -------
    tuple[dict[str, Any], SearchReport]
        Best parameters and a full search report.
    """
    optuna = _import_optuna()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info(
        "hpo_start",
        model_type=model_type,
        n_trials=n_trials,
        metric=metric,
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=f"creditguard_{model_type}",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    objective = _make_objective(X, y, model_type, n_cv_splits, random_state)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best_params = study.best_params
    best_score = study.best_value

    trial_history = [
        {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        }
        for t in study.trials
    ]

    report = SearchReport(
        model_type=model_type,
        n_trials=n_trials,
        best_score=best_score,
        best_params=best_params,
        metric=metric,
        trial_history=trial_history,
    )

    logger.info(
        "hpo_complete",
        model_type=model_type,
        best_score=round(best_score, 5),
        best_params=best_params,
    )
    return best_params, report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    import yaml

    parser = argparse.ArgumentParser(description="Run hyperparameter search")
    parser.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    features_path = Path(cfg["data"]["features_path"])
    target_col = cfg["data"]["target_column"]
    id_col = cfg["data"]["id_column"]

    if not features_path.exists():
        logger.error("features_not_found", path=str(features_path))
        raise SystemExit(1)

    df = pd.read_parquet(features_path)
    X = df.drop(columns=[target_col, id_col], errors="ignore")
    y = df[target_col]

    model_type = args.model or cfg["training"]["model"]
    n_trials = args.trials or cfg["training"]["hyperparameter_search"]["n_trials"]

    best_params, report = run_search(X, y, model_type=model_type, n_trials=n_trials)

    print(f"\nBest {model_type} params (PR-AUC={report.best_score:.5f}):")
    print(json.dumps(best_params, indent=2))
