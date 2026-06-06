"""Evaluate both models with honest OOF predictions and run champion-challenger comparison.

Loads trained model artifacts (feature_names + HPO params), re-creates CV
splits, trains per-fold to get OOF predictions, then runs the full evaluation
suite and governance comparison.

Outputs:
    artifacts/evaluation_results.json          -- champion/challenger metrics
    artifacts/oof_predictions_scored.parquet   -- champion OOF scores + sensitive
        attributes, consumed by scripts/fit_calibrator.py and
        scripts/reconcile_fairness.py (and the Streamlit dashboard).

Usage:
    python -m scripts.evaluate_and_compare
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import yaml

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer(colors=True)],
    wrapper_class=structlog.make_filtering_bound_logger(0),
)
log = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_artifact(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_model(model_type: str, params: dict):
    """Instantiate a fresh model with given params."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier

    clean = {k: v for k, v in params.items() if v is not None}
    if model_type == "lightgbm":
        return lgb.LGBMClassifier(verbose=-1, n_jobs=-1, random_state=42, **clean)
    elif model_type == "xgboost":
        return xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
            random_state=42,
            **clean,
        )
    elif model_type == "rf":
        return RandomForestClassifier(n_jobs=-1, random_state=42, **clean)
    raise ValueError(f"Unknown model type: {model_type}")


def get_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    params: dict,
    id_col_values: pd.Series,
    n_splits: int = 5,
) -> np.ndarray:
    """Train per-fold and collect honest OOF predictions."""
    from training.cv_pipeline import get_cv_splits

    # Build a minimal df for the split function
    split_df = pd.DataFrame({"SK_ID_CURR": id_col_values, "TARGET": y})
    splits, _ = get_cv_splits(
        split_df,
        strategy="stratified_group",
        n_splits=n_splits,
        target_col="TARGET",
        group_col="SK_ID_CURR",
    )

    oof_scores = np.zeros(len(y))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]

        model = build_model(model_type, params)
        model.fit(X_train, y_train)
        oof_scores[val_idx] = model.predict_proba(X_val)[:, 1]

        log.info("fold_oof", fold=fold_idx, model=model_type)

    return oof_scores


def evaluate_model(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    n_features: int,
    sensitive: pd.Series | None = None,
    decision_threshold: float = 0.40,
):
    """Run full evaluation suite and return ModelMetrics.

    If ``sensitive`` is provided, the fairness block runs against the real
    protected attribute (typically CODE_GENDER, joined from the unfeaturised
    table). If not, all three fairness fields (adverse_impact_ratio,
    demographic_parity_diff, equalized_odds_diff) are set to
    ``float('nan')`` so downstream governance code cannot silently
    treat an unmeasured result as a passing result. See
    ``governance/FAIRNESS_REPORT.md``.
    """
    from evaluation.calibration import compute_calibration_report
    from evaluation.discrimination import compute_discrimination_report
    from evaluation.fairness import compute_fairness_report
    from evaluation.threshold_analysis import compute_threshold_report
    from monitoring.champion_challenger import ModelMetrics

    disc = compute_discrimination_report(y_true, y_score)
    cal = compute_calibration_report(y_true, y_score)
    thresh = compute_threshold_report(y_true, y_score, fp_cost=600, fn_cost=6000)

    if sensitive is not None:
        # Positional alignment guard. This block applies a mask derived from
        # ``sensitive`` to ``y_true`` and ``y_score`` by position. ``.values``
        # strips any pandas index so positional alignment is only correct
        # when all three arrays share the same row order. A caller that
        # filters ``sensitive`` upstream without re-aligning ``y_true`` /
        # ``y_score`` would silently produce a fairness report against the
        # wrong rows — raise loudly instead.
        if not (len(sensitive) == len(y_true) == len(y_score)):
            raise ValueError(
                f"length mismatch in evaluate_model: sensitive={len(sensitive)}, "
                f"y_true={len(y_true)}, y_score={len(y_score)}"
            )
        mask = pd.Series(sensitive).notna().values
        y_pred = (np.asarray(y_score) >= decision_threshold).astype(int)
        fair_report = compute_fairness_report(
            np.asarray(y_true)[mask],
            y_pred[mask],
            np.asarray(y_score)[mask],
            pd.Series(sensitive)[mask].reset_index(drop=True),
        )
        air = fair_report.adverse_impact_ratio
        dp_vals = list(fair_report.demographic_parity.values())
        dp_diff = max(dp_vals) - min(dp_vals) if dp_vals else 0.0
        eo_tpr = [g["tpr"] for g in fair_report.equalized_odds.values()]
        eo_fpr = [g["fpr"] for g in fair_report.equalized_odds.values()]
        eo_diff = max(
            (max(eo_tpr) - min(eo_tpr)) if eo_tpr else 0.0,
            (max(eo_fpr) - min(eo_fpr)) if eo_fpr else 0.0,
        )
    else:
        # Sensitive attribute not provided; do NOT default to a passing value.
        # Use NaN so downstream governance code never silently treats
        # "unmeasured" as "passing 80% rule".
        air = float("nan")
        dp_diff = float("nan")
        eo_diff = float("nan")

    metrics = ModelMetrics(
        model_name=model_name,
        roc_auc=disc.roc_auc,
        pr_auc=disc.pr_auc,
        brier_score=cal.brier_score,
        ks_statistic=disc.ks_statistic,
        gini=disc.gini,
        calibration_gap=cal.ece,
        adverse_impact_ratio=air,
        demographic_parity_diff=dp_diff,
        equalized_odds_diff=eo_diff,
        interpretability_tier="medium",  # tree models
        n_features=n_features,
    )

    return metrics, disc, cal, thresh


def main():
    log.info("loading config and data")

    # Load config
    config_path = PROJECT_ROOT / "config" / "train.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load features
    features_path = PROJECT_ROOT / cfg["data"]["features_path"]
    df = pd.read_parquet(features_path)
    target_col = cfg["data"]["target_column"]
    id_col = cfg["data"]["id_column"]
    y = df[target_col]
    id_values = df[id_col]

    # Join sensitive attributes from the unfeaturised join table for the
    # fairness block. CODE_GENDER is the sensitive column used by default;
    # NAME_EDUCATION_TYPE is available in ``joined.parquet`` for a finer
    # slice. The ``data/processed/features.parquet`` file intentionally
    # excludes these (Boruta + VIF dropped them), so the sensitive join
    # happens here rather than in the feature matrix.
    joined_path = PROJECT_ROOT / "data" / "processed" / "joined.parquet"
    sensitive_series: pd.Series | None = None
    education_series: pd.Series | None = None
    if joined_path.exists():
        try:
            joined = pd.read_parquet(
                joined_path, columns=[id_col, "CODE_GENDER", "NAME_EDUCATION_TYPE"]
            )
            merged = df[[id_col]].merge(joined, on=id_col, how="left")
            # XNA is a Home Credit sentinel, not a real group — treat as NaN
            sensitive_series = merged["CODE_GENDER"].replace({"XNA": np.nan})
            education_series = merged["NAME_EDUCATION_TYPE"]
            log.info(
                "sensitive_joined",
                columns=["CODE_GENDER", "NAME_EDUCATION_TYPE"],
                non_null=int(sensitive_series.notna().sum()),
            )
        except Exception as exc:
            log.warning("sensitive_join_failed", error=str(exc))

    # Load model artifacts. The champion (LightGBM) is required — fail loudly
    # if it is missing. The challenger (XGBoost) is optional: if its pickle
    # has not been produced (a fresh clone only ran `make train` against the
    # default config), skip it gracefully and emit a champion-only report.
    # Matches the skip-when-missing pattern in scripts/fit_calibrator.py.
    lgb_path = PROJECT_ROOT / "artifacts" / "lightgbm_model.pkl"
    xgb_path = PROJECT_ROOT / "artifacts" / "xgboost_model.pkl"
    if not lgb_path.exists():
        raise FileNotFoundError(f"Champion artifact missing at {lgb_path}. Run `make train` first.")
    lgb_artifact = load_artifact(lgb_path)

    artifacts_to_score: list[tuple[str, dict]] = [("lightgbm", lgb_artifact)]
    if xgb_path.exists():
        artifacts_to_score.append(("xgboost", load_artifact(xgb_path)))
    else:
        log.warning(
            "challenger_artifact_missing_skipped",
            path=str(xgb_path),
            hint="train a challenger with model:xgboost in config to populate this",
        )

    results = {}

    for name, artifact in artifacts_to_score:
        model_type = artifact["model_type"]
        feature_names = artifact["feature_names"]
        params = artifact["params"]
        X = df[feature_names]

        log.info("getting_oof_predictions", model=name, n_features=len(feature_names))
        oof_scores = get_oof_predictions(X, y, model_type, params, id_values)

        log.info("evaluating", model=name)
        metrics, disc, cal, thresh = evaluate_model(
            y.values,
            oof_scores,
            model_name=name,
            n_features=len(feature_names),
            # Passes CODE_GENDER only. NAME_EDUCATION_TYPE fairness (AIR 0.6056)
            # is measured separately in scripts/reconcile_fairness.py and is
            # not included in the metrics dict written to evaluation_results.json.
            sensitive=sensitive_series,
        )

        results[name] = {
            "metrics": metrics,
            "discrimination": disc,
            "calibration": cal,
            "threshold": thresh,
            "oof_scores": oof_scores,
        }

        log.info(
            "model_evaluated",
            model=name,
            roc_auc=round(disc.roc_auc, 5),
            pr_auc=round(disc.pr_auc, 5),
            ks=round(disc.ks_statistic, 5),
            gini=round(disc.gini, 5),
            brier=round(cal.brier_score, 6),
            ece=round(cal.ece, 6),
            optimal_threshold=round(thresh.optimal_threshold, 4),
        )

    # Champion-challenger comparison
    from monitoring.champion_challenger import compare_models, promote_decision

    lgb_metrics = results["lightgbm"]["metrics"]

    if "xgboost" in results:
        xgb_metrics = results["xgboost"]["metrics"]
        report = compare_models(lgb_metrics, xgb_metrics)
        decision = promote_decision(report)
    else:
        xgb_metrics = None
        report = None
        decision = None

    # Print results
    print("\n" + "=" * 70)
    print("CREDITGUARD MODEL EVALUATION RESULTS")
    print("=" * 70)

    for name in [n for n in ["lightgbm", "xgboost"] if n in results]:
        m = results[name]["metrics"]
        d = results[name]["discrimination"]
        c = results[name]["calibration"]
        t = results[name]["threshold"]
        print(f"\n{'-' * 70}")
        print(f"  {name.upper()} ({'Champion' if name == 'lightgbm' else 'Challenger'})")
        print(f"{'-' * 70}")
        print(f"  ROC-AUC:            {m.roc_auc:.5f}")
        print(f"  PR-AUC:             {m.pr_auc:.5f}")
        print(f"  KS Statistic:       {m.ks_statistic:.5f}")
        print(f"  Gini:               {m.gini:.5f}")
        print(f"  Brier Score:        {m.brier_score:.6f}")
        print(f"  ECE:                {c.ece:.6f}")
        print(f"  Platt Brier:        {c.platt_brier:.6f}")
        print(f"  Isotonic Brier:     {c.isotonic_brier:.6f}")
        print(f"  Optimal Threshold:  {t.optimal_threshold:.4f}")
        print(f"  Cost at Optimal:    ${t.expected_cost_at_optimal:.2f}")
        print(f"  Features:           {m.n_features}")
        lift = d.lift_at_k
        print(f"  Lift@1%:            {lift.get('top_1pct', 0):.2f}x")
        print(f"  Lift@5%:            {lift.get('top_5pct', 0):.2f}x")
        print(f"  Lift@10%:           {lift.get('top_10pct', 0):.2f}x")

    if report is not None:
        print(f"\n{'=' * 70}")
        print("CHAMPION vs CHALLENGER COMPARISON")
        print("=" * 70)
        print(report.comparison_table.to_string(index=False))
        print("\nGovernance Notes:")
        for note in report.notes:
            print(f"  - {note}")
        print(f"\nPROMOTION DECISION: {decision.upper()}")
        print("=" * 70)
    else:
        print("\nNo challenger artifact found; champion-only evaluation complete.")

    # Save results to JSON
    output: dict = {
        "champion": {
            "model": "lightgbm",
            "metrics": lgb_metrics.to_dict(),
            "discrimination": {
                "roc_auc": results["lightgbm"]["discrimination"].roc_auc,
                "pr_auc": results["lightgbm"]["discrimination"].pr_auc,
                "ks_statistic": results["lightgbm"]["discrimination"].ks_statistic,
                "gini": results["lightgbm"]["discrimination"].gini,
                "lift_at_k": results["lightgbm"]["discrimination"].lift_at_k,
            },
            "calibration": {
                "brier_score": results["lightgbm"]["calibration"].brier_score,
                "ece": results["lightgbm"]["calibration"].ece,
                "platt_brier": results["lightgbm"]["calibration"].platt_brier,
                "isotonic_brier": results["lightgbm"]["calibration"].isotonic_brier,
            },
            "threshold": {
                "optimal_threshold": results["lightgbm"]["threshold"].optimal_threshold,
                "expected_cost_at_optimal": results["lightgbm"][
                    "threshold"
                ].expected_cost_at_optimal,
                "review_queue": results["lightgbm"]["threshold"].review_queue,
            },
        },
    }
    if "xgboost" in results:
        output["challenger"] = {
            "model": "xgboost",
            "metrics": xgb_metrics.to_dict(),
            "discrimination": {
                "roc_auc": results["xgboost"]["discrimination"].roc_auc,
                "pr_auc": results["xgboost"]["discrimination"].pr_auc,
                "ks_statistic": results["xgboost"]["discrimination"].ks_statistic,
                "gini": results["xgboost"]["discrimination"].gini,
                "lift_at_k": results["xgboost"]["discrimination"].lift_at_k,
            },
            "calibration": {
                "brier_score": results["xgboost"]["calibration"].brier_score,
                "ece": results["xgboost"]["calibration"].ece,
                "platt_brier": results["xgboost"]["calibration"].platt_brier,
                "isotonic_brier": results["xgboost"]["calibration"].isotonic_brier,
            },
            "threshold": {
                "optimal_threshold": results["xgboost"]["threshold"].optimal_threshold,
                "expected_cost_at_optimal": results["xgboost"][
                    "threshold"
                ].expected_cost_at_optimal,
                "review_queue": results["xgboost"]["threshold"].review_queue,
            },
        }
        output["comparison"] = {
            "pr_auc_improvement": report.pr_auc_improvement,
            "calibration_degradation": report.calibration_degradation,
            "fairness_within_tolerance": report.fairness_within_tolerance,
            "interpretability_acceptable": report.interpretability_acceptable,
            "notes": report.notes,
            "decision": decision,
        }

    output_path = PROJECT_ROOT / "artifacts" / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("results_saved", path=str(output_path))

    # Persist the champion's honest OOF predictions so the downstream
    # calibration + fairness steps (scripts/fit_calibrator.py,
    # scripts/reconcile_fairness.py) and the Streamlit dashboard operate on
    # scores consistent with the model just evaluated. Without this write the
    # `make evaluate` chain would silently reuse a stale committed parquet.
    champion_oof = pd.DataFrame(
        {
            id_col: np.asarray(id_values),
            "y_true": np.asarray(y),
            "y_score": np.asarray(results["lightgbm"]["oof_scores"]),
        }
    )
    if sensitive_series is not None:
        champion_oof["CODE_GENDER"] = np.asarray(sensitive_series)
    if education_series is not None:
        champion_oof["NAME_EDUCATION_TYPE"] = np.asarray(education_series)

    oof_path = PROJECT_ROOT / "artifacts" / "oof_predictions_scored.parquet"
    champion_oof.to_parquet(oof_path, index=False)
    log.info("oof_predictions_saved", path=str(oof_path), n_rows=len(champion_oof))


if __name__ == "__main__":
    main()
