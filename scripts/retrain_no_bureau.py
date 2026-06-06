"""Bureau-score ablation: retrain LightGBM without any EXT_SOURCE_* features.

Measures the discrimination loss when every bureau-derived external-score
feature (raw EXT_SOURCE_*, pairwise interactions, mean, std) is removed
from the training data.

Rationale:
- The champion's top-10 mean |SHAP| is dominated by five EXT features that
  contribute ~61% of the attribution on a log-odds basis
  (see features/FEATURE_IMPORTANCE.md).
- In a production deployment, bureau-score columns are contractually
  licensed and may be unavailable for segments (thin-file, new-credit
  markets, bureau outage). Quantifying the lift from our engineered
  non-bureau features is a business-critical planning number.

Design:
- Re-uses the champion's tuned hyperparameters rather than re-running
  Optuna HPO; this isolates the feature-set effect from HPO noise and
  keeps the run reproducible in under an hour.
- Writes a new artifact at artifacts/lightgbm_no_bureau.pkl. The
  champion artifact (artifacts/lightgbm_model.pkl) is never touched.
- Uses the same 5-fold StratifiedGroupKFold (grouped on SK_ID_CURR) as
  the champion evaluation, so OOF metrics are directly comparable.

Usage:
    python -m scripts.retrain_no_bureau
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer(colors=True)],
    wrapper_class=structlog.make_filtering_bound_logger(0),
)
log = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_lightgbm(params: dict):
    import lightgbm as lgb

    clean = {k: v for k, v in params.items() if v is not None}
    return lgb.LGBMClassifier(verbose=-1, n_jobs=-1, random_state=42, **clean)


def filter_bureau(feature_names: list[str], exclude_prefixes: list[str]) -> list[str]:
    """Drop any feature matching any exclude_prefix."""
    keep = [f for f in feature_names if not any(f.startswith(pref) for pref in exclude_prefixes)]
    dropped = sorted(set(feature_names) - set(keep))
    log.info("ablation_filter", kept=len(keep), dropped=len(dropped), dropped_features=dropped)
    return keep


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic between default and non-default score distributions."""
    from scipy.stats import ks_2samp

    s_good = y_score[y_true == 0]
    s_bad = y_score[y_true == 1]
    return float(ks_2samp(s_good, s_bad).statistic)


def main():
    t_start = time.time()

    # ── Load config ──────────────────────────────────────
    config_path = PROJECT_ROOT / "config" / "train_no_bureau.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    log.info("config_loaded", path=str(config_path))

    # ── Load data ────────────────────────────────────────
    features_path = PROJECT_ROOT / cfg["data"]["features_path"]
    df = pd.read_parquet(features_path)
    target_col = cfg["data"]["target_column"]
    id_col = cfg["data"]["id_column"]
    y = df[target_col]
    id_values = df[id_col]

    log.info("data_loaded", n_rows=len(df), pos_rate=round(float(y.mean()), 4))

    # ── Load champion to borrow HPs and its already-selected feature set ──
    champion_path = PROJECT_ROOT / "artifacts" / "lightgbm_model.pkl"
    with open(champion_path, "rb") as f:
        champion = pickle.load(f)

    champion_features = champion["feature_names"]
    params = champion["params"]
    log.info("champion_loaded", n_features=len(champion_features), params=params)

    # ── Apply ablation filter ────────────────────────────
    exclude_prefixes = cfg["feature_selection"].get("exclude_prefixes", [])
    feature_names = filter_bureau(champion_features, exclude_prefixes)

    if len(feature_names) == len(champion_features):
        log.warning(
            "no_features_filtered",
            hint="check exclude_prefixes in config/train_no_bureau.yaml",
        )

    X = df[feature_names]
    log.info("ablation_set", n_features=len(feature_names), features=feature_names)

    # ── CV splits (same 5-fold StratifiedGroupKFold as champion) ──
    from training.cv_pipeline import get_cv_splits

    split_df = pd.DataFrame({id_col: id_values, target_col: y})
    splits, cv_report = get_cv_splits(
        split_df,
        strategy="stratified_group",
        n_splits=cfg["training"]["cv"]["n_splits"],
        target_col=target_col,
        group_col=id_col,
    )

    # ── Per-fold OOF ─────────────────────────────────────
    oof_scores = np.zeros(len(y))
    fold_roc_aucs = []
    fold_pr_aucs = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = build_lightgbm(params)
        model.fit(X_train, y_train)

        val_scores = model.predict_proba(X_val)[:, 1]
        oof_scores[val_idx] = val_scores

        fold_roc = roc_auc_score(y_val, val_scores)
        fold_pr = average_precision_score(y_val, val_scores)
        fold_roc_aucs.append(fold_roc)
        fold_pr_aucs.append(fold_pr)
        log.info("fold_done", fold=fold_idx, roc_auc=round(fold_roc, 5), pr_auc=round(fold_pr, 5))

    # ── OOF aggregate metrics ────────────────────────────
    oof_roc = float(roc_auc_score(y, oof_scores))
    oof_pr = float(average_precision_score(y, oof_scores))
    oof_ks = ks_statistic(y.values, oof_scores)

    log.info(
        "oof_summary",
        oof_roc_auc=round(oof_roc, 5),
        oof_pr_auc=round(oof_pr, 5),
        oof_ks=round(oof_ks, 5),
        fold_roc_mean=round(float(np.mean(fold_roc_aucs)), 5),
        fold_roc_std=round(float(np.std(fold_roc_aucs)), 5),
    )

    # ── Retrain on full data and persist ─────────────────
    final_model = build_lightgbm(params)
    final_model.fit(X, y)

    artifact_dir = PROJECT_ROOT / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / "lightgbm_no_bureau.pkl"

    artifact = {
        "model": final_model,
        "scaler": None,
        "feature_names": feature_names,
        "params": params,
        "model_type": "lightgbm",
        "ablation": {
            "excluded_prefixes": exclude_prefixes,
            "dropped_from_champion": sorted(set(champion_features) - set(feature_names)),
            "oof_roc_auc": oof_roc,
            "oof_pr_auc": oof_pr,
            "oof_ks": oof_ks,
        },
    }
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)

    log.info("artifact_saved", path=str(out_path))

    # ── Emit a JSON summary for the doc write-up ─────────
    summary = {
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "artifact": str(out_path.relative_to(PROJECT_ROOT)),
        "n_features_champion": len(champion_features),
        "n_features_ablation": len(feature_names),
        "excluded_features": artifact["ablation"]["dropped_from_champion"],
        "oof": {
            "roc_auc": oof_roc,
            "pr_auc": oof_pr,
            "ks": oof_ks,
        },
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    summary_path = artifact_dir / "ablation_no_bureau_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("BUREAU-SCORE ABLATION SUMMARY")
    print("=" * 70)
    print(f"  Features: {len(champion_features)} -> {len(feature_names)}")
    print(f"  Dropped:  {artifact['ablation']['dropped_from_champion']}")
    print(f"  OOF ROC-AUC: {oof_roc:.5f}")
    print(f"  OOF PR-AUC:  {oof_pr:.5f}")
    print(f"  OOF KS:      {oof_ks:.5f}")
    print(f"  Elapsed:     {summary['elapsed_seconds']}s")
    print(f"  Summary:     {summary_path.relative_to(PROJECT_ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
