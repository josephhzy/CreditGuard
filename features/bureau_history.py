"""Bureau history features from bureau and bureau_balance tables.

Captures the applicant's credit history as reported by external credit
bureaus: number of past credits, overdue statistics, active-vs-closed
mix, and credit type distribution.

Missing-value strategy:
  - Aggregated counts default to 0 when no bureau records exist (genuinely
    no external credit history).
  - Statistical summaries (mean, max) use -999 sentinel when there are no
    records to aggregate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

SENTINEL = -999


def build_bureau_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features from bureau credit history.

    Parameters
    ----------
    df : pd.DataFrame
        Joined dataset with one row per SK_ID_CURR. Expected to contain
        aggregated bureau columns (prefixed ``bureau_`` or ``buro_``) or
        raw bureau columns from the join.

    Returns
    -------
    pd.DataFrame
        New feature columns indexed like *df*.
    """
    out = pd.DataFrame(index=df.index)

    # ── Credit counts ───────────────────────────────────────────────
    # The join layer typically aggregates these; we look for common column
    # names from both raw and pre-aggregated patterns.
    active_col = _find_col(
        df, ["bureau_CREDIT_ACTIVE_Active", "CREDIT_ACTIVE_Active", "buro_CREDIT_ACTIVE_Active"]
    )
    closed_col = _find_col(
        df, ["bureau_CREDIT_ACTIVE_Closed", "CREDIT_ACTIVE_Closed", "buro_CREDIT_ACTIVE_Closed"]
    )

    active_count = df[active_col].fillna(0) if active_col else pd.Series(0, index=df.index)
    closed_count = df[closed_col].fillna(0) if closed_col else pd.Series(0, index=df.index)

    out["feat_bureau_active_count"] = active_count
    out["feat_bureau_closed_count"] = closed_count
    out["feat_bureau_total_count"] = active_count + closed_count
    out["feat_bureau_active_ratio"] = np.where(
        (active_count + closed_count) > 0,
        active_count / (active_count + closed_count),
        SENTINEL,
    )

    # ── Overdue statistics ──────────────────────────────────────────
    overdue_col = _find_col(
        df, ["CREDIT_DAY_OVERDUE", "bureau_CREDIT_DAY_OVERDUE_mean", "buro_CREDIT_DAY_OVERDUE_mean"]
    )
    if overdue_col:
        out["feat_bureau_overdue_mean"] = df[overdue_col].fillna(SENTINEL)
    else:
        out["feat_bureau_overdue_mean"] = SENTINEL

    overdue_max_col = _find_col(
        df,
        [
            "AMT_CREDIT_MAX_OVERDUE",
            "bureau_AMT_CREDIT_MAX_OVERDUE_max",
            "buro_AMT_CREDIT_MAX_OVERDUE_max",
        ],
    )
    if overdue_max_col:
        out["feat_bureau_max_overdue_amt"] = df[overdue_max_col].fillna(SENTINEL)
    else:
        out["feat_bureau_max_overdue_amt"] = SENTINEL

    # ── Recency of adverse events ───────────────────────────────────
    # Days since most recent overdue (smaller = more recent = riskier)
    dpd_col = _find_col(df, ["bureau_AMT_CREDIT_MAX_OVERDUE_max", "CREDIT_DAY_OVERDUE"])
    if dpd_col:
        has_overdue = df[dpd_col].gt(0)
        out["feat_bureau_any_overdue_flag"] = has_overdue.astype(int)
    else:
        out["feat_bureau_any_overdue_flag"] = 0

    # ── Bureau credit amount ────────────────────────────────────────
    amt_col = _find_col(
        df, ["bureau_AMT_CREDIT_SUM_mean", "AMT_CREDIT_SUM", "buro_AMT_CREDIT_SUM_mean"]
    )
    if amt_col:
        out["feat_bureau_credit_amt_mean"] = df[amt_col].fillna(SENTINEL)
    else:
        out["feat_bureau_credit_amt_mean"] = SENTINEL

    amt_sum_col = _find_col(df, ["bureau_AMT_CREDIT_SUM_sum", "buro_AMT_CREDIT_SUM_sum"])
    if amt_sum_col:
        out["feat_bureau_credit_amt_sum"] = df[amt_sum_col].fillna(SENTINEL)
    else:
        out["feat_bureau_credit_amt_sum"] = SENTINEL

    # ── Credit type distribution ────────────────────────────────────
    credit_type_cols = [c for c in df.columns if "CREDIT_TYPE" in c and c != "CREDIT_TYPE"]
    for col in credit_type_cols[:10]:  # cap to avoid explosion
        safe_name = col.replace(" ", "_").replace("/", "_")
        out[f"feat_bureau_{safe_name}"] = df[col].fillna(0)

    # ── Bureau balance DPD stats (if pre-aggregated) ────────────────
    bb_dpd_col = _find_col(
        df, ["bureau_balance_STATUS_DPD_mean", "bb_STATUS_DPD_mean", "bbal_DPD_mean"]
    )
    if bb_dpd_col:
        out["feat_bb_dpd_mean"] = df[bb_dpd_col].fillna(SENTINEL)

    n_features = len(out.columns)
    log.info("bureau_history features built", n_features=n_features)
    return out


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name found in *df*, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None
