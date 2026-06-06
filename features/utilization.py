"""Utilization features from credit_card_balance and POS_CASH_balance.

Measures how aggressively the applicant uses revolving credit (cards)
and point-of-sale / cash credit products: utilization ratios, balance
trends, cash withdrawal intensity, and POS DPD patterns.

Missing-value strategy:
  - Utilization ratios use -999 sentinel when credit limit is zero or missing.
  - Counts default to 0 (no card/POS records = no usage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

SENTINEL = -999


def build_utilization_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build credit utilization features.

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

    # ── Card utilization ratio ──────────────────────────────────────
    bal_col = _find_col(
        df,
        [
            "cc_AMT_BALANCE_mean",
            "credit_card_AMT_BALANCE_mean",
            "cc_AMT_BALANCE_max",
        ],
    )
    limit_col = _find_col(
        df,
        [
            "cc_AMT_CREDIT_LIMIT_ACTUAL_mean",
            "credit_card_AMT_CREDIT_LIMIT_ACTUAL_mean",
            "cc_AMT_CREDIT_LIMIT_ACTUAL_max",
        ],
    )

    if bal_col and limit_col:
        out["feat_cc_utilization_mean"] = np.where(
            df[limit_col].gt(0),
            df[bal_col] / df[limit_col],
            SENTINEL,
        )
    else:
        out["feat_cc_utilization_mean"] = SENTINEL

    # ── Peak utilization ────────────────────────────────────────────
    bal_max_col = _find_col(
        df,
        [
            "cc_AMT_BALANCE_max",
            "credit_card_AMT_BALANCE_max",
        ],
    )
    if bal_max_col and limit_col:
        out["feat_cc_peak_utilization"] = np.where(
            df[limit_col].gt(0),
            df[bal_max_col] / df[limit_col],
            SENTINEL,
        )
    else:
        out["feat_cc_peak_utilization"] = SENTINEL

    # ── Average balance (level) ─────────────────────────────────────
    if bal_col:
        out["feat_cc_avg_balance"] = df[bal_col].fillna(SENTINEL)
    else:
        out["feat_cc_avg_balance"] = SENTINEL

    # ── Balance trend (slope proxy) ─────────────────────────────────
    # If aggregation provides first-month and last-month balances, compute slope.
    # Otherwise, use max minus min as a range proxy.
    bal_min_col = _find_col(
        df,
        [
            "cc_AMT_BALANCE_min",
            "credit_card_AMT_BALANCE_min",
        ],
    )
    if bal_max_col and bal_min_col:
        out["feat_cc_balance_range"] = np.where(
            df[bal_max_col].notna() & df[bal_min_col].notna(),
            df[bal_max_col] - df[bal_min_col],
            SENTINEL,
        )
    else:
        out["feat_cc_balance_range"] = SENTINEL

    # ── Cash withdrawal intensity ───────────────────────────────────
    atm_col = _find_col(
        df,
        [
            "cc_AMT_DRAWINGS_ATM_CURRENT_mean",
            "credit_card_AMT_DRAWINGS_ATM_CURRENT_mean",
            "cc_AMT_DRAWINGS_ATM_CURRENT_sum",
        ],
    )
    if atm_col:
        out["feat_cc_atm_drawings_mean"] = df[atm_col].fillna(0)
    else:
        out["feat_cc_atm_drawings_mean"] = 0

    atm_count_col = _find_col(
        df,
        [
            "cc_CNT_DRAWINGS_ATM_CURRENT_sum",
            "credit_card_CNT_DRAWINGS_ATM_CURRENT_sum",
            "cc_CNT_DRAWINGS_ATM_CURRENT_mean",
        ],
    )
    if atm_count_col:
        out["feat_cc_atm_drawings_count"] = df[atm_count_col].fillna(0)
    else:
        out["feat_cc_atm_drawings_count"] = 0

    # ── POS cash DPD patterns ───────────────────────────────────────
    pos_dpd_mean = _find_col(
        df,
        [
            "pos_SK_DPD_mean",
            "POS_CASH_SK_DPD_mean",
            "pos_SK_DPD_DEF_mean",
            "POS_CASH_SK_DPD_DEF_mean",
        ],
    )
    pos_dpd_max = _find_col(
        df,
        [
            "pos_SK_DPD_max",
            "POS_CASH_SK_DPD_max",
            "pos_SK_DPD_DEF_max",
            "POS_CASH_SK_DPD_DEF_max",
        ],
    )

    if pos_dpd_mean:
        out["feat_pos_dpd_mean"] = df[pos_dpd_mean].fillna(0)
    else:
        out["feat_pos_dpd_mean"] = 0

    if pos_dpd_max:
        out["feat_pos_dpd_max"] = df[pos_dpd_max].fillna(0)
    else:
        out["feat_pos_dpd_max"] = 0

    # Any POS DPD flag
    if pos_dpd_max:
        out["feat_pos_any_dpd_flag"] = df[pos_dpd_max].gt(0).astype(int)
    else:
        out["feat_pos_any_dpd_flag"] = 0

    # ── Active months count ─────────────────────────────────────────
    active_col = _find_col(
        df,
        [
            "cc_MONTHS_BALANCE_count",
            "credit_card_MONTHS_BALANCE_count",
            "cc_count",
        ],
    )
    if active_col:
        out["feat_cc_active_months"] = df[active_col].fillna(0)
    else:
        out["feat_cc_active_months"] = 0

    pos_count_col = _find_col(
        df,
        [
            "pos_MONTHS_BALANCE_count",
            "POS_CASH_MONTHS_BALANCE_count",
            "pos_count",
        ],
    )
    if pos_count_col:
        out["feat_pos_active_months"] = df[pos_count_col].fillna(0)
    else:
        out["feat_pos_active_months"] = 0

    n_features = len(out.columns)
    log.info("utilization features built", n_features=n_features)
    return out


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name found in *df*, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None
