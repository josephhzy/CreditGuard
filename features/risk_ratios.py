"""Derived risk ratio features combining multiple data sources.

These are the highest-signal engineered features: debt burden ratios,
annuity-to-income, credit-to-income, payment-to-income, and external
source interaction terms. Ratios are powerful because they normalise
absolute amounts by the applicant's capacity.

Missing-value strategy:
  - Division by zero or missing denominator yields -999 sentinel.
  - External source columns are notoriously sparse; missing combinations
    are filled with -999 to let tree-based models split on them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

SENTINEL = -999


def build_risk_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build ratio features that combine information across tables.

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
    income = df["AMT_INCOME_TOTAL"]
    has_income = income.gt(0) & income.notna()

    # ── Debt burden ratio ───────────────────────────────────────────
    out["feat_debt_burden"] = np.where(has_income, df["AMT_CREDIT"] / income, SENTINEL)

    # ── Annuity-to-income ───────────────────────────────────────────
    out["feat_annuity_to_income"] = np.where(
        has_income, df["AMT_ANNUITY"].fillna(0) / income, SENTINEL
    )

    # ── Goods-to-income ─────────────────────────────────────────────
    out["feat_goods_to_income"] = np.where(
        has_income, df["AMT_GOODS_PRICE"].fillna(0) / income, SENTINEL
    )

    # ── Bureau credit-to-income ─────────────────────────────────────
    bureau_sum_col = _find_col(
        df,
        [
            "bureau_AMT_CREDIT_SUM_sum",
            "buro_AMT_CREDIT_SUM_sum",
            "bureau_AMT_CREDIT_SUM_mean",
        ],
    )
    if bureau_sum_col:
        out["feat_bureau_credit_to_income"] = np.where(
            has_income, df[bureau_sum_col].fillna(0) / income, SENTINEL
        )
    else:
        out["feat_bureau_credit_to_income"] = SENTINEL

    # ── Payment-to-income ───────────────────────────────────────────
    pay_col = _find_col(
        df,
        [
            "inst_AMT_PAYMENT_mean",
            "instal_AMT_PAYMENT_mean",
            "inst_AMT_PAYMENT_sum",
        ],
    )
    if pay_col:
        out["feat_payment_to_income"] = np.where(
            has_income, df[pay_col].fillna(0) / income, SENTINEL
        )
    else:
        out["feat_payment_to_income"] = SENTINEL

    # ── Annuity-to-credit ───────────────────────────────────────────
    has_credit = df["AMT_CREDIT"].gt(0) & df["AMT_CREDIT"].notna()
    out["feat_annuity_to_credit"] = np.where(
        has_credit, df["AMT_ANNUITY"].fillna(0) / df["AMT_CREDIT"], SENTINEL
    )

    # NOTE: credit-to-goods ratio is built in application_profile.py
    # to avoid duplicate feature names across families.

    # ── External source features ────────────────────────────────────
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    present = [c for c in ext_cols if c in df.columns]

    for c in present:
        out[f"feat_{c.lower()}"] = df[c].fillna(SENTINEL)

    # Pairwise interaction terms
    if "EXT_SOURCE_1" in df.columns and "EXT_SOURCE_2" in df.columns:
        out["feat_ext_1x2"] = np.where(
            df["EXT_SOURCE_1"].notna() & df["EXT_SOURCE_2"].notna(),
            df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"],
            SENTINEL,
        )
    if "EXT_SOURCE_1" in df.columns and "EXT_SOURCE_3" in df.columns:
        out["feat_ext_1x3"] = np.where(
            df["EXT_SOURCE_1"].notna() & df["EXT_SOURCE_3"].notna(),
            df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"],
            SENTINEL,
        )
    if "EXT_SOURCE_2" in df.columns and "EXT_SOURCE_3" in df.columns:
        out["feat_ext_2x3"] = np.where(
            df["EXT_SOURCE_2"].notna() & df["EXT_SOURCE_3"].notna(),
            df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"],
            SENTINEL,
        )

    # External source mean and std across the three sources
    if present:
        ext_df = df[present]
        out["feat_ext_source_mean"] = ext_df.mean(axis=1).fillna(SENTINEL)
        # std requires >= 2 non-null values to be meaningful
        out["feat_ext_source_std"] = np.where(
            ext_df.notna().sum(axis=1) >= 2,
            ext_df.std(axis=1),
            SENTINEL,
        )
        out["feat_ext_source_nancount"] = ext_df.isna().sum(axis=1)
    else:
        out["feat_ext_source_mean"] = SENTINEL
        out["feat_ext_source_std"] = SENTINEL
        out["feat_ext_source_nancount"] = 3

    n_features = len(out.columns)
    log.info("risk_ratios features built", n_features=n_features)
    return out


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name found in *df*, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None
