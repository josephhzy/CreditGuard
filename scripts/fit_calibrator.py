"""Fit Platt-scaling calibrators on OOF predictions and patch model artifacts.

The training script saves a model bundle ``{"model", "scaler",
"feature_names", "params", "model_type"}``. The serving stack then expects
``risk_score`` to be a calibrated PD (per ``governance/model_card.md`` and
the ``README.md`` field semantics). This script reads
``artifacts/oof_predictions_scored.parquet`` (honest out-of-fold scores),
fits a logistic regression on ``y_score → y_true`` for each model, and
writes the fitted calibrator back into the bundle under the ``calibrator``
and ``calibrator_method`` keys.

Run this whenever ``train.py`` and ``scripts/evaluate_and_compare.py`` have
both been re-executed and the OOF parquet has been refreshed.

Usage:
    python -m scripts.fit_calibrator
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import structlog
from sklearn.linear_model import LogisticRegression

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer(colors=True)],
    wrapper_class=structlog.make_filtering_bound_logger(0),
)
log = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OOF_PATH = PROJECT_ROOT / "artifacts" / "oof_predictions_scored.parquet"


def fit_platt(y_score: pd.Series, y_true: pd.Series) -> LogisticRegression:
    """Fit a 1-D logistic regression mapping raw scores to calibrated PDs.

    Platt scaling fits ``P(y=1 | s) = sigmoid(a*s + b)`` on the OOF scores.
    Sklearn's ``LogisticRegression`` on a single-column feature gives this
    directly. We use no regularisation (large ``C``) because Platt's original
    formulation is unregularised maximum likelihood.
    """
    model = LogisticRegression(C=1e9, solver="lbfgs", max_iter=200)
    model.fit(y_score.to_numpy().reshape(-1, 1), y_true.to_numpy())
    return model


def patch_artifact(artifact_path: Path, calibrator: LogisticRegression) -> None:
    """Load a model bundle, attach the calibrator, and write it back."""
    with open(artifact_path, "rb") as f:
        bundle = pickle.load(f)  # noqa: S301 — internal artifact

    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Unexpected artifact shape at {artifact_path}: {type(bundle).__name__}")

    bundle["calibrator"] = calibrator
    bundle["calibrator_method"] = "platt_logistic_on_oof"

    with open(artifact_path, "wb") as f:
        pickle.dump(bundle, f)
    log.info("artifact_patched", path=str(artifact_path))


def main() -> None:
    if not OOF_PATH.exists():
        raise FileNotFoundError(
            f"OOF predictions not found at {OOF_PATH}. Run scripts/evaluate_and_compare.py first."
        )

    df = pd.read_parquet(OOF_PATH)
    log.info("oof_loaded", n_rows=len(df), default_rate=round(float(df["y_true"].mean()), 4))

    # Per-model OOF predictions are stored in the same parquet under a single
    # ``y_score`` column for the champion. If you re-run the evaluator with
    # multiple models writing to separate columns, extend the loop accordingly.
    calibrator = fit_platt(df["y_score"], df["y_true"])

    # Sanity check: calibrated mean should match the empirical default rate.
    cal_mean = float(calibrator.predict_proba(df["y_score"].to_numpy().reshape(-1, 1))[:, 1].mean())
    log.info(
        "calibrator_fit",
        intercept=round(float(calibrator.intercept_[0]), 4),
        slope=round(float(calibrator.coef_[0][0]), 4),
        raw_mean=round(float(df["y_score"].mean()), 4),
        calibrated_mean=round(cal_mean, 4),
        empirical_default_rate=round(float(df["y_true"].mean()), 4),
    )

    # Only patch the champion. The challenger (XGBoost) has no persisted OOF
    # scores, so attaching the LightGBM calibrator to it would be misleading.
    for name in ("lightgbm_model.pkl",):
        path = PROJECT_ROOT / "artifacts" / name
        if path.exists():
            patch_artifact(path, calibrator)
        else:
            log.warning("artifact_missing_skipped", path=str(path))


if __name__ == "__main__":
    main()
