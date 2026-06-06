"""Adverse-impact ratio (AIR) sweep across a grid of decline thresholds.

Context:
- The prior fairness round (see governance/FAIRNESS_REPORT.md) measured
  AIR at three operational thresholds: 0.15 (approve gate), 0.40 (decline
  gate), and 0.50 (cost-optimum). AIR failed the 80% rule at t=0.15 for
  CODE_GENDER (0.71) and at t=0.40 for NAME_EDUCATION_TYPE (0.61).
- To localise where the 80% rule starts to bind, this script evaluates
  AIR at the full grid [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
  0.50] for both sensitive attributes.

Data source:
- artifacts/oof_predictions_scored.parquet contains SK_ID_CURR, y_true,
  y_score (OOF champion score), and the two sensitive columns.
- XNA values in CODE_GENDER are treated as NaN and excluded.

Output:
- artifacts/fairness_threshold_sweep.csv

Usage:
    python -m scripts.fairness_threshold_sweep
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from evaluation.fairness import adverse_impact_ratio

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer(colors=True)],
    wrapper_class=structlog.make_filtering_bound_logger(0),
)
log = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
SENSITIVE_COLS = ["CODE_GENDER", "NAME_EDUCATION_TYPE"]


def compute_air_at_threshold(
    y_score: np.ndarray,
    sensitive: pd.Series,
    threshold: float,
) -> float:
    """AIR at a given decline threshold.

    y_pred = 1 when score >= threshold (decline).
    approval_rate_g = share of group predicted 0 (approved).
    AIR = min_g approval_rate_g / max_g approval_rate_g.

    Rows with NaN in `sensitive` are dropped.
    """
    mask = sensitive.notna().values
    yp = (y_score[mask] >= threshold).astype(int)
    return adverse_impact_ratio(yp, sensitive[mask].values)


def approval_rates_at_threshold(
    y_score: np.ndarray,
    sensitive: pd.Series,
    threshold: float,
) -> dict[str, float]:
    """Per-group approval rate for transparency in the output table."""
    mask = sensitive.notna().values
    yp = (y_score[mask] >= threshold).astype(int)
    s = sensitive[mask].values
    out = {}
    for g in sorted(set(s.tolist())):
        gm = s == g
        if gm.sum() == 0:
            continue
        out[str(g)] = float((yp[gm] == 0).sum() / gm.sum())
    return out


def main():
    # ── Load OOF scores + sensitive join ────────────────
    oof_path = PROJECT_ROOT / "artifacts" / "oof_predictions_scored.parquet"
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF predictions not found at {oof_path}")

    df = pd.read_parquet(oof_path)
    log.info("oof_loaded", rows=len(df), cols=df.columns.tolist())

    # Normalise CODE_GENDER: XNA is a Home Credit sentinel, not a real group
    df["CODE_GENDER"] = df["CODE_GENDER"].replace({"XNA": np.nan})

    y_score = df["y_score"].values

    rows = []
    for t in THRESHOLDS:
        row = {"threshold": t}
        for col in SENSITIVE_COLS:
            sensitive = df[col]
            air = compute_air_at_threshold(y_score, sensitive, t)
            rates = approval_rates_at_threshold(y_score, sensitive, t)
            n_valid = int(sensitive.notna().sum())
            row[f"AIR_{col}"] = round(air, 4)
            row[f"pass_80pct_{col}"] = "yes" if air >= 0.80 else "no"
            row[f"n_valid_{col}"] = n_valid
            row[f"min_approval_{col}"] = round(min(rates.values()), 4) if rates else None
            row[f"max_approval_{col}"] = round(max(rates.values()), 4) if rates else None
            log.info(
                "air_at_threshold",
                threshold=t,
                sensitive=col,
                air=round(air, 4),
                passes=air >= 0.80,
            )
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "artifacts" / "fairness_threshold_sweep.csv"
    out.to_csv(out_path, index=False)
    log.info("saved", path=str(out_path))

    # Compact console table
    print("\n" + "=" * 90)
    print("FAIRNESS AIR SWEEP — champion LightGBM OOF (307,511 applicants)")
    print("=" * 90)
    print(
        out[
            [
                "threshold",
                "AIR_CODE_GENDER",
                "pass_80pct_CODE_GENDER",
                "AIR_NAME_EDUCATION_TYPE",
                "pass_80pct_NAME_EDUCATION_TYPE",
            ]
        ].to_string(index=False)
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
