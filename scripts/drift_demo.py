"""Synthetic drift demonstration for CreditGuard.

Perturbs ONE feature by a caller-specified multiple of its reference-period
standard deviation, then reports PSI per feature plus the combined drift
report from :mod:`monitoring.drift_engine`. The unshifted 14 features stay
green while the shifted feature walks from green -> amber -> red as
``--shift-sigma`` grows.

Usage
-----
.. code-block:: bash

    python -m scripts.drift_demo --feature feat_debt_burden --shift-sigma 0.5
    python -m scripts.drift_demo --feature feat_debt_burden --shift-sigma 1.0
    python -m scripts.drift_demo --feature feat_ext_source_2 --shift-sigma 2.0

See ``monitoring/DRIFT_DEMO.md`` for the methodology and the expected
shift-vs-PSI table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from monitoring.drift_engine import detect_drift
from monitoring.psi import classify_drift, psi_per_feature

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The 15 features the champion LightGBM was trained on. Source of truth is
# ``artifacts/lightgbm_model.pkl`` (field ``feature_names``);
# inlined here so the demo runs without loading the pickle.
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic drift demonstration on the 15 selected features.",
    )
    parser.add_argument(
        "--feature",
        default="feat_debt_burden",
        choices=SELECTED_FEATURES,
        help="Feature to perturb (default: feat_debt_burden).",
    )
    parser.add_argument(
        "--shift-sigma",
        type=float,
        default=1.0,
        help="Shift the target feature by this many reference-period stddevs.",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "features.parquet",
        help="Parquet file produced by ``features/build_all.py``.",
    )
    parser.add_argument(
        "--ref-fraction",
        type=float,
        default=0.8,
        help="Fraction of rows to put into the reference period (default: 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the reference/current split (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.features_path.exists():
        raise SystemExit(
            f"Features parquet not found at {args.features_path}. Run ``make features`` first.",
        )

    df = pd.read_parquet(args.features_path)
    missing = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing:
        raise SystemExit(f"Missing expected features in parquet: {missing}")

    # Random 80/20 reference/current split. The aggregated feature matrix
    # does not carry a clean application-date column, so a temporal split is
    # out of scope here; see ``monitoring/DRIFT_DEMO.md`` section 7.
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(df))
    cutoff = int(args.ref_fraction * len(df))
    ref = df.iloc[idx[:cutoff]][SELECTED_FEATURES].copy()
    cur = df.iloc[idx[cutoff:]][SELECTED_FEATURES].copy()

    # Apply synthetic drift in units of the reference-period stddev.
    sigma = float(ref[args.feature].std())
    if not np.isfinite(sigma) or sigma == 0.0:
        raise SystemExit(
            f"Reference stddev for {args.feature} is {sigma}; cannot shift.",
        )
    cur[args.feature] = cur[args.feature] + args.shift_sigma * sigma

    print(
        f"Demo: --feature {args.feature}  --shift-sigma {args.shift_sigma}  "
        f"(ref n={len(ref):,}, cur n={len(cur):,}, ref_sigma={sigma:.4f})",
    )
    print("-" * 78)

    # PSI per feature.
    psi_values = psi_per_feature(
        ref.values,
        cur.values,
        feature_names=SELECTED_FEATURES,
    )
    print("PSI per feature (fixed 10-bin on reference):")
    for f in SELECTED_FEATURES:
        v = psi_values[f]
        label = classify_drift(v)
        marker = "  <-- target" if f == args.feature else ""
        print(f"  {f:50s} PSI={v:.4f}  {label}{marker}")

    # Combined drift report (PSI + KS + JS).
    report = detect_drift(ref, cur, feature_names=SELECTED_FEATURES)
    print()
    print(
        f"Overall severity: {report.overall_severity}  "
        f"(green={report.n_features_green}, "
        f"amber={report.n_features_amber}, red={report.n_features_red})",
    )
    top_names = set(report.top_drifting_features)
    print(f"Top drifters ({len(top_names)}):")
    for fd in report.per_feature:
        if fd.feature in top_names:
            print(
                f"  {fd.feature:50s} PSI={fd.psi:.4f}  "
                f"KS={fd.ks_statistic:.4f} (p={fd.ks_pvalue:.4g})  "
                f"JS={fd.js_divergence:.4f}",
            )
    if report.root_cause_notes:
        print("Root cause notes:")
        for note in report.root_cause_notes:
            print(f"  - {note}")


if __name__ == "__main__":
    main()
