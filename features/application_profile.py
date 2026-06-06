"""Application profile features from the main application table.

Extracts demographic, financial, employment, and documentation features
from the primary application record. These are the most directly available
predictors since they come from the application itself.

Missing-value strategy:
  - Numeric features: filled with -999 sentinel to preserve missingness signal
    for tree-based models. The sentinel is outside the natural range of all
    features, so trees can split on it without contaminating real distributions.
  - Categorical ordinals: filled with -1 (unknown category).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

SENTINEL = -999

# Ordinal mapping for education level (lower = less formal education)
_EDUCATION_ORDINAL: dict[str, int] = {
    "Lower secondary": 0,
    "Secondary / secondary special": 1,
    "Incomplete higher": 2,
    "Higher education": 3,
    "Academic degree": 4,
}


def build_application_features(
    df: pd.DataFrame,
    *,
    income_quantile_edges: list[float] | None = None,
) -> pd.DataFrame:
    """Build features derived from the main application table.

    Parameters
    ----------
    df : pd.DataFrame
        Joined dataset with one row per SK_ID_CURR. Expected to contain
        columns from the ``application_{train|test}.csv`` table.
    income_quantile_edges : list[float] | None, optional
        Pre-computed bin edges for ``feat_income_quintile``, produced by
        ``pd.qcut(..., retbins=True)[1]`` on training rows only.  When
        supplied, ``pd.cut`` is used with these fixed edges so no
        distribution information from the current ``df`` is incorporated
        (safe inside a CV fold).  When ``None`` (default), edges are
        derived from ``df`` itself via ``pd.qcut`` — only appropriate when
        ``df`` is the full dataset (e.g. the single call in
        ``features/build_all.py``).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed like *df* containing only the new feature columns.
    """
    out = pd.DataFrame(index=df.index)

    # ── Income features ─────────────────────────────────────────────
    out["feat_income"] = df["AMT_INCOME_TOTAL"].fillna(SENTINEL)
    out["feat_log_income"] = np.where(
        df["AMT_INCOME_TOTAL"].gt(0),
        np.log1p(df["AMT_INCOME_TOTAL"]),
        SENTINEL,
    )
    if income_quantile_edges is not None:
        # CV-safe path: use pre-fit edges from training rows only.
        out["feat_income_quintile"] = (
            pd.cut(
                df["AMT_INCOME_TOTAL"],
                bins=income_quantile_edges,
                labels=False,
                include_lowest=True,
            )
            .fillna(SENTINEL)
            .astype(float)
        )
    else:
        # Full-dataset call (build_all.py): derive edges from all rows.
        _quintile_series, _edges = pd.qcut(
            df["AMT_INCOME_TOTAL"],
            q=5,
            labels=False,
            duplicates="drop",
            retbins=True,
        )
        out["feat_income_quintile"] = _quintile_series.fillna(SENTINEL).astype(float)

    # ── Credit features ─────────────────────────────────────────────
    out["feat_credit_amount"] = df["AMT_CREDIT"].fillna(SENTINEL)
    out["feat_annuity"] = df["AMT_ANNUITY"].fillna(SENTINEL)
    out["feat_goods_price"] = df["AMT_GOODS_PRICE"].fillna(SENTINEL)
    out["feat_credit_to_goods"] = np.where(
        df["AMT_GOODS_PRICE"].gt(0),
        df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"],
        SENTINEL,
    )

    # ── Employment features ─────────────────────────────────────────
    # DAYS_EMPLOYED is negative (days before application); 365243 = unemployed sentinel in raw data
    days_emp = df["DAYS_EMPLOYED"].copy()
    employed_flag = days_emp.ne(365243) & days_emp.notna()
    days_emp = days_emp.where(employed_flag, np.nan)

    out["feat_employed_flag"] = employed_flag.astype(int)
    out["feat_employment_years"] = np.where(
        employed_flag,
        (-days_emp / 365.25).round(2),
        SENTINEL,
    )
    # Employment stability: ratio of employment duration to applicant age
    age_years = -df["DAYS_BIRTH"] / 365.25
    out["feat_employment_stability"] = np.where(
        employed_flag & age_years.gt(0),
        (-days_emp / 365.25) / age_years,
        SENTINEL,
    )

    # ── Family features ─────────────────────────────────────────────
    out["feat_children_count"] = df["CNT_CHILDREN"].fillna(SENTINEL)
    out["feat_family_size"] = df["CNT_FAM_MEMBERS"].fillna(SENTINEL)
    out["feat_family_to_income"] = np.where(
        df["AMT_INCOME_TOTAL"].gt(0) & df["CNT_FAM_MEMBERS"].gt(0),
        df["CNT_FAM_MEMBERS"] / df["AMT_INCOME_TOTAL"],
        SENTINEL,
    )

    # ── Education (ordinal encode) ──────────────────────────────────
    out["feat_education_ordinal"] = (
        df["NAME_EDUCATION_TYPE"].map(_EDUCATION_ORDINAL).fillna(-1).astype(int)
    )

    # ── Housing (one-hot) ───────────────────────────────────────────
    if "NAME_HOUSING_TYPE" in df.columns:
        housing_dummies = pd.get_dummies(df["NAME_HOUSING_TYPE"], prefix="feat_housing", dtype=int)
        # Sanitise column names (spaces, slashes)
        housing_dummies.columns = [
            c.replace(" ", "_").replace("/", "_") for c in housing_dummies.columns
        ]
        out = pd.concat([out, housing_dummies], axis=1)

    # ── Age features ────────────────────────────────────────────────
    out["feat_age_years"] = np.where(
        df["DAYS_BIRTH"].notna(),
        (-df["DAYS_BIRTH"] / 365.25).round(2),
        SENTINEL,
    )
    out["feat_age_bin"] = (
        pd.cut(
            -df["DAYS_BIRTH"] / 365.25,
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=False,
        )
        .fillna(SENTINEL)
        .astype(float)
    )

    # ── Document flags ──────────────────────────────────────────────
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        out["feat_document_count"] = df[doc_cols].sum(axis=1)
    else:
        out["feat_document_count"] = SENTINEL

    n_features = len(out.columns)
    log.info("application_profile features built", n_features=n_features)
    return out
