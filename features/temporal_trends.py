"""Temporal trend features capturing behavioural trajectory.

These features detect *direction of change* -- whether an applicant's
credit behaviour is worsening or improving. A single snapshot of DPD
is less informative than whether DPD is trending upward. Trends are
estimated via simple linear regression slopes over time-series data
that the data layer has aggregated.

Missing-value strategy:
  - Slope features use -999 sentinel when fewer than 2 data points exist
    (a slope cannot be estimated from a single observation).
  - Binary flags default to 0 (no evidence of worsening).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

SENTINEL = -999


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build trend features that capture behavioural trajectory.

    Parameters
    ----------
    df : pd.DataFrame
        Joined dataset with one row per SK_ID_CURR. May contain
        pre-computed slope columns from the data layer, or raw
        recent-vs-old aggregations that we derive slopes from.

    Returns
    -------
    pd.DataFrame
        New feature columns indexed like *df*.
    """
    out = pd.DataFrame(index=df.index)

    # ── Delinquency trend ───────────────────────────────────────────
    # If aggregation provides recent and old DPD means, compute a delta.
    dpd_recent = _find_col(
        df,
        [
            "inst_dpd_last_3",
            "instal_dpd_last_3",
            "inst_late_last_3",
            "instal_late_last_3",
            "inst_payment_late_mean_last_6m",
        ],
    )
    dpd_old = _find_col(
        df,
        [
            "inst_dpd_last_12",
            "instal_dpd_last_12",
            "inst_late_last_12",
            "instal_late_last_12",
            "inst_payment_late_mean",
        ],
    )

    if dpd_recent and dpd_old:
        recent_val = df[dpd_recent].fillna(0)
        old_val = df[dpd_old].fillna(0)
        out["feat_trend_dpd_delta"] = recent_val - old_val
        out["feat_trend_dpd_worsening"] = (out["feat_trend_dpd_delta"] > 0).astype(int)
    else:
        out["feat_trend_dpd_delta"] = SENTINEL
        out["feat_trend_dpd_worsening"] = 0

    # ── Utilization trend ───────────────────────────────────────────
    # Proxy: difference between max balance and mean balance
    # (peak utilization higher than average suggests increasing usage)
    cc_bal_mean = _find_col(
        df,
        [
            "cc_AMT_BALANCE_mean",
            "credit_card_AMT_BALANCE_mean",
        ],
    )
    cc_bal_max = _find_col(
        df,
        [
            "cc_AMT_BALANCE_max",
            "credit_card_AMT_BALANCE_max",
        ],
    )

    if cc_bal_mean and cc_bal_max:
        out["feat_trend_util_delta"] = np.where(
            df[cc_bal_mean].notna() & df[cc_bal_max].notna(),
            df[cc_bal_max] - df[cc_bal_mean],
            SENTINEL,
        )
    else:
        out["feat_trend_util_delta"] = SENTINEL

    # ── Balance velocity ────────────────────────────────────────────
    # Rate of balance change: (max_balance - min_balance) / n_months
    cc_bal_min = _find_col(
        df,
        [
            "cc_AMT_BALANCE_min",
            "credit_card_AMT_BALANCE_min",
        ],
    )
    cc_months = _find_col(
        df,
        [
            "cc_MONTHS_BALANCE_count",
            "credit_card_MONTHS_BALANCE_count",
            "cc_count",
        ],
    )

    if cc_bal_max and cc_bal_min and cc_months:
        n_months = df[cc_months].fillna(0)
        out["feat_trend_balance_velocity"] = np.where(
            n_months > 1,
            (df[cc_bal_max].fillna(0) - df[cc_bal_min].fillna(0)) / n_months,
            SENTINEL,
        )
    else:
        out["feat_trend_balance_velocity"] = SENTINEL

    # ── Payment timeliness trend ────────────────────────────────────
    # Compare recent payment behaviour to historical
    pay_recent = _find_col(
        df,
        [
            "inst_AMT_PAYMENT_mean_last_6m",
            "instal_AMT_PAYMENT_mean_last_6m",
        ],
    )
    pay_hist = _find_col(
        df,
        [
            "inst_AMT_PAYMENT_mean",
            "instal_AMT_PAYMENT_mean",
        ],
    )

    if pay_recent and pay_hist:
        out["feat_trend_payment_delta"] = np.where(
            df[pay_hist].gt(0),
            (df[pay_recent].fillna(0) - df[pay_hist].fillna(0)) / df[pay_hist],
            SENTINEL,
        )
    else:
        out["feat_trend_payment_delta"] = SENTINEL

    # ── POS DPD trend ───────────────────────────────────────────────
    pos_dpd_mean = _find_col(
        df,
        [
            "pos_SK_DPD_mean",
            "POS_CASH_SK_DPD_mean",
        ],
    )
    pos_dpd_max = _find_col(
        df,
        [
            "pos_SK_DPD_max",
            "POS_CASH_SK_DPD_max",
        ],
    )

    if pos_dpd_mean and pos_dpd_max:
        out["feat_trend_pos_dpd_spike"] = np.where(
            df[pos_dpd_mean].notna(),
            df[pos_dpd_max].fillna(0) - df[pos_dpd_mean].fillna(0),
            SENTINEL,
        )
    else:
        out["feat_trend_pos_dpd_spike"] = SENTINEL

    # ── Composite worsening flag ────────────────────────────────────
    # Flag if multiple indicators suggest deterioration
    worsening_signals = pd.DataFrame(index=df.index)
    if "feat_trend_dpd_worsening" in out.columns:
        worsening_signals["dpd"] = out["feat_trend_dpd_worsening"]
    if "feat_trend_util_delta" in out.columns:
        worsening_signals["util"] = (
            out["feat_trend_util_delta"].ne(SENTINEL) & out["feat_trend_util_delta"].gt(0)
        ).astype(int)
    if "feat_trend_pos_dpd_spike" in out.columns:
        worsening_signals["pos"] = (
            out["feat_trend_pos_dpd_spike"].ne(SENTINEL) & out["feat_trend_pos_dpd_spike"].gt(0)
        ).astype(int)

    if len(worsening_signals.columns) > 0:
        out["feat_trend_worsening_score"] = worsening_signals.sum(axis=1)
    else:
        out["feat_trend_worsening_score"] = 0

    n_features = len(out.columns)
    log.info("temporal_trends features built", n_features=n_features)
    return out


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name found in *df*, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None
