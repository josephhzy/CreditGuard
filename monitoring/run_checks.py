"""Generate ``monitoring/reports/latest_drift.json``.

This is the producer that the API's ``/drift/report`` endpoint reads from. It
runs the existing PSI + KS + JS engine over a reference / current split and
writes a JSON report shaped for ``serving/routes/drift.py``.

Two operating modes:

1. **Full-feature mode** (``--features data/processed/features.parquet``).
   Reads the engineered feature matrix produced by ``features.build_all`` and
   reports drift across the 15 selected features. This is what the docs and
   ``make monitor`` describe.
2. **Demo / minimum-artifacts mode** (default if features.parquet is absent).
   Falls back to the OOF score parquet that ships with the repo and reports
   single-column drift on ``y_score``. The output report is clearly labelled
   ``demo: true`` so callers can tell.

In both modes the current period is the second half of a deterministic
shuffle of the input rows. Set ``--shift-feature`` to inject a synthetic
sigma shift for repeatable demos (matches ``scripts/drift_demo.py``).

Usage::

    python -m monitoring.run_checks
    python -m monitoring.run_checks --features data/processed/features.parquet
    python -m monitoring.run_checks --shift-feature feat_debt_burden --shift-sigma 1.5
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from monitoring.psi import calculate_psi, classify_drift, psi_per_feature

logger = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
DEFAULT_OOF_PATH = PROJECT_ROOT / "artifacts" / "oof_predictions_scored.parquet"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "monitoring" / "reports" / "latest_drift.json"

SELECTED_FEATURES: list[str] = [
    "feat_annuity",
    "feat_credit_to_goods",
    "feat_employment_years",
    "feat_education_ordinal",
    "feat_age_bin",
    "feat_document_count",
    "feat_bureau_BUREAU_CREDIT_TYPE_nunique",
    "feat_prev_PREV_NAME_CASH_LOAN_PURPOSE_nunique",
    "feat_debt_burden",
    "feat_annuity_to_credit",
    "feat_ext_source_2",
    "feat_ext_1x2",
    "feat_ext_2x3",
    "feat_ext_source_mean",
    "feat_ext_source_std",
]


def _split_ref_current(
    df: pd.DataFrame, ref_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cutoff = int(ref_fraction * len(df))
    return df.iloc[idx[:cutoff]].copy(), df.iloc[idx[cutoff:]].copy()


def _apply_shift(
    cur: pd.DataFrame, feature: str, shift_sigma: float, ref: pd.DataFrame
) -> pd.DataFrame:
    if feature not in cur.columns:
        raise SystemExit(f"--shift-feature {feature} not found in input columns.")
    sigma = float(ref[feature].std())
    if not np.isfinite(sigma) or sigma == 0.0:
        raise SystemExit(f"Reference stddev for {feature} is {sigma}; cannot shift.")
    cur[feature] = cur[feature] + shift_sigma * sigma
    return cur


def _feature_record(name: str, psi: float, threshold: float) -> dict[str, Any]:
    severity = classify_drift(psi).upper()
    return {
        "feature": name,
        "method": "psi",
        "statistic": round(float(psi), 6),
        "threshold": threshold,
        "drifted": bool(psi >= threshold),
        "severity": severity,
    }


def _build_report(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    feature_names: list[str],
    *,
    threshold: float,
    demo: bool,
    notes: list[str],
) -> dict[str, Any]:
    psi_values = psi_per_feature(
        ref[feature_names].values,
        cur[feature_names].values,
        feature_names=feature_names,
    )
    features = [_feature_record(name, psi_values[name], threshold) for name in feature_names]

    return {
        "report_date": datetime.now(timezone.utc).date().isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "demo": demo,
        "notes": notes,
        "n_reference_rows": int(len(ref)),
        "n_current_rows": int(len(cur)),
        "alert_threshold": threshold,
        "features": features,
    }


def _generate_full_feature_report(args: argparse.Namespace) -> dict[str, Any]:
    df = pd.read_parquet(args.features)
    missing = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing:
        raise SystemExit(f"Missing expected features in {args.features}: {missing}")

    ref, cur = _split_ref_current(df[SELECTED_FEATURES], args.ref_fraction, args.seed)
    notes = [
        f"reference/current split: {args.ref_fraction:.2f}/{1 - args.ref_fraction:.2f}",
        "reference and current are a random 80/20 split of the same historical data — not a temporal split.",
    ]
    if args.shift_feature:
        cur = _apply_shift(cur, args.shift_feature, args.shift_sigma, ref)
        notes.append(f"synthetic shift: {args.shift_feature} += {args.shift_sigma}σ")
    return _build_report(
        ref,
        cur,
        SELECTED_FEATURES,
        threshold=args.alert_threshold,
        demo=bool(args.shift_feature),
        notes=notes,
    )


def _generate_score_report(args: argparse.Namespace) -> dict[str, Any]:
    df = pd.read_parquet(args.oof)
    if "y_score" not in df.columns:
        raise SystemExit(f"OOF parquet at {args.oof} has no 'y_score' column.")
    ref, cur = _split_ref_current(df[["y_score"]], args.ref_fraction, args.seed)

    # Apply a synthetic shift only when explicitly requested. The earlier
    # behaviour injected 0.5σ unconditionally so the demo report would be
    # visibly non-zero, but that fabricated drift on a freshly-installed
    # environment and the same report feeds /drift/report. y_score is a
    # probability so the shifted values are clipped to [0, 1] to preserve
    # PD semantics any downstream consumer assumes.
    if args.shift_feature:
        cur = _apply_shift(cur, "y_score", args.shift_sigma, ref)
        cur["y_score"] = cur["y_score"].clip(0.0, 1.0)
        shift_note = f"synthetic shift: y_score += {args.shift_sigma:.2f}σ (clipped to [0,1])."
    else:
        shift_note = "no synthetic shift; reference and current are exchangeable halves (PSI ≈ 0)."

    psi = calculate_psi(ref["y_score"].values, cur["y_score"].values)
    notes = [
        "demo report: full feature matrix not present, drift computed on y_score only.",
        shift_note,
        "Run `make features` then `make monitor` for a real multi-feature report.",
    ]
    return {
        "report_date": datetime.now(timezone.utc).date().isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "demo": True,
        "notes": notes,
        "n_reference_rows": int(len(ref)),
        "n_current_rows": int(len(cur)),
        "alert_threshold": args.alert_threshold,
        "features": [_feature_record("y_score", psi, args.alert_threshold)],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Engineered features parquet (full-feature mode).",
    )
    parser.add_argument(
        "--oof",
        type=Path,
        default=DEFAULT_OOF_PATH,
        help="OOF predictions parquet (demo fallback).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Where to write the JSON report.",
    )
    parser.add_argument("--ref-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shift-feature", default=None)
    parser.add_argument("--shift-sigma", type=float, default=1.0)
    parser.add_argument("--alert-threshold", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.features.exists():
        report = _generate_full_feature_report(args)
        logger.info("monitor_run", mode="full_feature", n_features=len(report["features"]))
    else:
        if not args.oof.exists():
            raise SystemExit(
                f"Neither {args.features} nor {args.oof} exists. "
                "Run `make features` or `make evaluate` first."
            )
        report = _generate_score_report(args)
        logger.info("monitor_run", mode="demo_score_only", n_features=len(report["features"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("drift_report_written", path=str(args.out), demo=report["demo"])


if __name__ == "__main__":
    main()
