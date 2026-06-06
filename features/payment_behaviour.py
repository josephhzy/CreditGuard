"""Payment behaviour features from installments_payments.

Captures how reliably the applicant makes installment payments:
lateness patterns, missed payments, rolling delinquency windows,
payment consistency, and underpayment ratios.

Missing-value strategy:
  - Count/flag features default to 0 (no installment records = no late payments).
  - Statistical features (mean, std) use -999 sentinel when no records exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

SENTINEL = -999


def build_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features from installment payment history.

    Parameters
    ----------
    df : pd.DataFrame
        Joined dataset with one row per SK_ID_CURR. Expected to contain
        pre-aggregated installment columns (e.g. ``inst_*`` or ``instal_*``).

    Returns
    -------
    pd.DataFrame
        New feature columns indexed like *df*.
    """
    out = pd.DataFrame(index=df.index)

    # ── Payment lateness ────────────────────────────────────────────
    # lateness = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT (positive = late)
    late_mean_col = _find_col(
        df,
        [
            "inst_payment_late_mean",
            "instal_payment_late_mean",
            "inst_DAYS_ENTRY_PAYMENT_mean",
            "instal_DAYS_ENTRY_PAYMENT_mean",
        ],
    )
    late_max_col = _find_col(
        df,
        [
            "inst_payment_late_max",
            "instal_payment_late_max",
            "inst_DAYS_ENTRY_PAYMENT_max",
            "instal_DAYS_ENTRY_PAYMENT_max",
        ],
    )

    if late_mean_col:
        out["feat_pay_lateness_mean"] = df[late_mean_col].fillna(SENTINEL)
    else:
        out["feat_pay_lateness_mean"] = SENTINEL

    if late_max_col:
        out["feat_pay_lateness_max"] = df[late_max_col].fillna(SENTINEL)
    else:
        out["feat_pay_lateness_max"] = SENTINEL

    # ── Missed payment frequency ────────────────────────────────────
    missed_col = _find_col(
        df,
        [
            "inst_missed_count",
            "instal_missed_count",
            "inst_NUM_INSTALMENT_NUMBER_count",
            "instal_NUM_INSTALMENT_NUMBER_count",
        ],
    )
    if missed_col:
        out["feat_pay_missed_count"] = df[missed_col].fillna(0)
    else:
        out["feat_pay_missed_count"] = 0

    # ── Rolling delinquency windows ─────────────────────────────────
    # These columns are expected from a pre-aggregation step that
    # computes late-payment counts for the last 3, 6, 12 installments.
    for window in [3, 6, 12]:
        col = _find_col(
            df,
            [
                f"inst_late_last_{window}",
                f"instal_late_last_{window}",
                f"inst_dpd_last_{window}",
                f"instal_dpd_last_{window}",
            ],
        )
        if col:
            out[f"feat_pay_late_last_{window}"] = df[col].fillna(0)
        else:
            out[f"feat_pay_late_last_{window}"] = 0

    # ── Payment variance / consistency ──────────────────────────────
    pay_std_col = _find_col(
        df,
        [
            "inst_AMT_PAYMENT_std",
            "instal_AMT_PAYMENT_std",
        ],
    )
    pay_mean_col = _find_col(
        df,
        [
            "inst_AMT_PAYMENT_mean",
            "instal_AMT_PAYMENT_mean",
        ],
    )
    if pay_std_col:
        out["feat_pay_amount_std"] = df[pay_std_col].fillna(SENTINEL)
    else:
        out["feat_pay_amount_std"] = SENTINEL

    if pay_mean_col:
        out["feat_pay_amount_mean"] = df[pay_mean_col].fillna(SENTINEL)
    else:
        out["feat_pay_amount_mean"] = SENTINEL

    # Coefficient of variation (higher = less consistent)
    if pay_std_col and pay_mean_col:
        out["feat_pay_cv"] = np.where(
            df[pay_mean_col].gt(0),
            df[pay_std_col] / df[pay_mean_col],
            SENTINEL,
        )
    else:
        out["feat_pay_cv"] = SENTINEL

    # ── Underpayment ratio ──────────────────────────────────────────
    # AMT_PAYMENT vs AMT_INSTALMENT (ratio < 1 means underpaid)
    inst_amt_col = _find_col(
        df,
        [
            "inst_AMT_INSTALMENT_mean",
            "instal_AMT_INSTALMENT_mean",
        ],
    )
    if pay_mean_col and inst_amt_col:
        out["feat_pay_underpayment_ratio"] = np.where(
            df[inst_amt_col].gt(0),
            df[pay_mean_col] / df[inst_amt_col],
            SENTINEL,
        )
    else:
        out["feat_pay_underpayment_ratio"] = SENTINEL

    # Count of installment records (proxy for loan history length)
    count_col = _find_col(
        df,
        [
            "inst_NUM_INSTALMENT_NUMBER_count",
            "instal_NUM_INSTALMENT_NUMBER_count",
            "inst_count",
            "instal_count",
        ],
    )
    if count_col:
        out["feat_pay_instalment_count"] = df[count_col].fillna(0)
    else:
        out["feat_pay_instalment_count"] = 0

    n_features = len(out.columns)
    log.info("payment_behaviour features built", n_features=n_features)
    return out


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name found in *df*, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None
