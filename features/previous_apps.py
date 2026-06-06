"""Previous application features.

Captures the applicant's history of prior loan applications at the
institution: approval/rejection patterns, requested amounts, timing
between applications, purpose mix, and yield-group distribution.

Missing-value strategy:
  - Counts default to 0 (no prior applications is genuine information).
  - Ratios and statistical summaries use -999 sentinel when the
    denominator is zero or no records exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

SENTINEL = -999


def build_previous_app_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features from the previous_application table.

    Parameters
    ----------
    df : pd.DataFrame
        Joined dataset with one row per SK_ID_CURR.

    Returns
    -------
    pd.DataFrame
        New feature columns indexed like *df*.
    """
    out = pd.DataFrame(index=df.index)

    # ── Approval / rejection counts ─────────────────────────────────
    approved_col = _find_col(
        df, ["prev_NAME_CONTRACT_STATUS_Approved", "previous_NAME_CONTRACT_STATUS_Approved"]
    )
    refused_col = _find_col(
        df, ["prev_NAME_CONTRACT_STATUS_Refused", "previous_NAME_CONTRACT_STATUS_Refused"]
    )
    canceled_col = _find_col(
        df, ["prev_NAME_CONTRACT_STATUS_Canceled", "previous_NAME_CONTRACT_STATUS_Canceled"]
    )

    approved = df[approved_col].fillna(0) if approved_col else pd.Series(0, index=df.index)
    refused = df[refused_col].fillna(0) if refused_col else pd.Series(0, index=df.index)
    canceled = df[canceled_col].fillna(0) if canceled_col else pd.Series(0, index=df.index)

    total_prev = approved + refused + canceled
    out["feat_prev_approved_count"] = approved
    out["feat_prev_refused_count"] = refused
    out["feat_prev_canceled_count"] = canceled
    out["feat_prev_total_count"] = total_prev
    out["feat_prev_approval_rate"] = np.where(total_prev > 0, approved / total_prev, SENTINEL)
    out["feat_prev_refusal_rate"] = np.where(total_prev > 0, refused / total_prev, SENTINEL)

    # ── Historical credit sizes ─────────────────────────────────────
    amt_mean_col = _find_col(df, ["prev_AMT_APPLICATION_mean", "previous_AMT_APPLICATION_mean"])
    amt_max_col = _find_col(df, ["prev_AMT_APPLICATION_max", "previous_AMT_APPLICATION_max"])

    out["feat_prev_amt_mean"] = (
        df[amt_mean_col].fillna(SENTINEL) if amt_mean_col else pd.Series(SENTINEL, index=df.index)
    )
    out["feat_prev_amt_max"] = (
        df[amt_max_col].fillna(SENTINEL) if amt_max_col else pd.Series(SENTINEL, index=df.index)
    )

    # Trend: max minus mean (larger spread = escalating requests)
    if amt_mean_col and amt_max_col:
        out["feat_prev_amt_spread"] = np.where(
            df[amt_mean_col].notna() & df[amt_max_col].notna(),
            df[amt_max_col] - df[amt_mean_col],
            SENTINEL,
        )
    else:
        out["feat_prev_amt_spread"] = SENTINEL

    # ── Application recency ─────────────────────────────────────────
    days_col = _find_col(
        df,
        [
            "prev_DAYS_DECISION_min",
            "previous_DAYS_DECISION_min",
            "prev_DAYS_DECISION_max",
            "previous_DAYS_DECISION_max",
        ],
    )
    if days_col:
        out["feat_prev_days_since_last"] = np.where(
            df[days_col].notna(),
            (-df[days_col]).round(0),
            SENTINEL,
        )
    else:
        out["feat_prev_days_since_last"] = SENTINEL

    # ── Repeated application pattern ────────────────────────────────
    # Already captured in total_prev above; add a "repeat applicant" flag
    out["feat_prev_repeat_flag"] = (total_prev > 1).astype(int)

    # ── Yield group distribution ────────────────────────────────────
    yield_cols = [c for c in df.columns if "NAME_YIELD_GROUP" in c or "YIELD_GROUP" in c]
    for col in yield_cols[:6]:
        safe_name = col.replace(" ", "_").replace("/", "_")
        out[f"feat_prev_{safe_name}"] = df[col].fillna(0)

    # ── Purpose mix ─────────────────────────────────────────────────
    purpose_cols = [
        c for c in df.columns if "NAME_CASH_LOAN_PURPOSE" in c or "CASH_LOAN_PURPOSE" in c
    ]
    for col in purpose_cols[:10]:
        safe_name = col.replace(" ", "_").replace("/", "_")
        out[f"feat_prev_{safe_name}"] = df[col].fillna(0)

    n_features = len(out.columns)
    log.info("previous_apps features built", n_features=n_features)
    return out


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name found in *df*, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None
