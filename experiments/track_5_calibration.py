"""Track 5 -- Calibration method comparison.

Compares probability calibration strategies:
  - Raw probabilities (no calibration)
  - Platt scaling (logistic / sigmoid)
  - Isotonic regression

Logs calibration curves and Brier scores to MLflow.

Usage::

    python -m experiments.track_5_calibration
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import structlog
import yaml
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from experiments.track_1_baseline import compute_metrics

if TYPE_CHECKING:
    import numpy as np

matplotlib.use("Agg")  # non-interactive backend for server/CI

logger = structlog.get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "train.yaml"


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def _plot_calibration_curves(
    results: list[dict[str, Any]],
    y_true: np.ndarray,
    save_path: Path,
    n_bins: int = 10,
) -> None:
    """Plot reliability diagrams for all calibration methods and save to disk."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    for entry in results:
        name = entry["method"]
        y_prob = entry["y_prob"]
        fraction_pos, mean_predicted = calibration_curve(y_true, y_prob, n_bins=n_bins)
        ax.plot(mean_predicted, fraction_pos, "s-", label=f"{name} (Brier={entry['brier']:.4f})")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("calibration_plot_saved", path=str(save_path))


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def main(config_path: Path | None = None) -> pd.DataFrame:
    """Run Track 5: calibration method comparison.

    Parameters
    ----------
    config_path : Path, optional
        Path to ``config/train.yaml``.

    Returns
    -------
    pd.DataFrame
        Summary table with Brier scores and discrimination metrics.
    """
    cfg_path = config_path or _DEFAULT_CONFIG
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    data_cfg = cfg["data"]
    calib_cfg = cfg.get("evaluation", {}).get("calibration", {})
    n_bins = calib_cfg.get("n_bins", 10)
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

    # Three-way split: train / calibrate / test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        stratify=y,
        random_state=seed,
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=seed,
    )

    logger.info(
        "data_split",
        n_train=len(X_train),
        n_cal=len(X_cal),
        n_test=len(X_test),
    )

    # -- Train base model on training set ------------------------------------
    base_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    base_model.fit(X_train, y_train)

    # -- MLflow setup --------------------------------------------------------
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns/"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "creditguard") + "/track_5_calibration")

    # -- Calibration variants ------------------------------------------------
    calibration_results: list[dict[str, Any]] = []

    # 1. Raw probabilities
    y_prob_raw = base_model.predict_proba(X_test)[:, 1]
    brier_raw = brier_score_loss(y_test, y_prob_raw)
    metrics_raw = compute_metrics(y_test, y_prob_raw)
    calibration_results.append(
        {
            "method": "raw",
            "y_prob": y_prob_raw,
            "brier": brier_raw,
            **metrics_raw,
        }
    )

    # 2. Platt scaling (sigmoid)
    platt_model = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
    platt_model.fit(X_cal, y_cal)
    y_prob_platt = platt_model.predict_proba(X_test)[:, 1]
    brier_platt = brier_score_loss(y_test, y_prob_platt)
    metrics_platt = compute_metrics(y_test, y_prob_platt)
    calibration_results.append(
        {
            "method": "platt",
            "y_prob": y_prob_platt,
            "brier": brier_platt,
            **metrics_platt,
        }
    )

    # 3. Isotonic regression
    iso_model = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
    iso_model.fit(X_cal, y_cal)
    y_prob_iso = iso_model.predict_proba(X_test)[:, 1]
    brier_iso = brier_score_loss(y_test, y_prob_iso)
    metrics_iso = compute_metrics(y_test, y_prob_iso)
    calibration_results.append(
        {
            "method": "isotonic",
            "y_prob": y_prob_iso,
            "brier": brier_iso,
            **metrics_iso,
        }
    )

    # -- Log to MLflow -------------------------------------------------------
    summary_rows: list[dict[str, Any]] = []

    for entry in calibration_results:
        name = entry["method"]
        with mlflow.start_run(run_name=name):
            mlflow.log_param("calibration_method", name)
            mlflow.log_param("seed", seed)
            mlflow.log_param("n_bins", n_bins)
            mlflow.log_metric("brier_score", round(entry["brier"], 6))

            for metric_name in ("roc_auc", "pr_auc", "ks_statistic", "gini"):
                mlflow.log_metric(metric_name, round(entry[metric_name], 6))

            summary_rows.append(
                {
                    "method": name,
                    "brier_score": entry["brier"],
                    "roc_auc": entry["roc_auc"],
                    "pr_auc": entry["pr_auc"],
                    "ks_statistic": entry["ks_statistic"],
                    "gini": entry["gini"],
                }
            )

    # Calibration curve plot as shared artifact
    plot_path = _PROJECT_ROOT / "mlflow" / "calibration_curves.png"
    _plot_calibration_curves(calibration_results, y_test, plot_path, n_bins=n_bins)

    # Log the plot under the last run (or a parent run)
    with mlflow.start_run(run_name="calibration_summary"):
        mlflow.log_artifact(str(plot_path))
        for row in summary_rows:
            mlflow.log_metric(f"brier_{row['method']}", round(row["brier_score"], 6))

    # -- Summary table -------------------------------------------------------
    summary = pd.DataFrame(summary_rows).sort_values("brier_score")
    print("\n" + "=" * 80)
    print("TRACK 5 -- CALIBRATION METHOD COMPARISON")
    print("=" * 80)
    print(summary.to_string(index=False, float_format="%.4f"))
    print("=" * 80 + "\n")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Track 5: Calibration method comparison")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    main(config_path=args.config)
