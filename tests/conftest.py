"""Shared pytest fixtures for CreditGuard test suite.

Provides reusable synthetic data generators so individual test modules
can focus on behaviour verification rather than data wrangling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Binary classification data
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_binary_data() -> tuple[np.ndarray, np.ndarray]:
    """500-row synthetic binary classification dataset (~8 % positive class).

    Returns (X, y) where X is (500, 20) float64 and y is (500,) int {0,1}.
    """
    rng = np.random.default_rng(42)
    n, p = 500, 20
    X = rng.standard_normal((n, p))
    # ~8 % positive rate
    y = rng.binomial(1, 0.08, size=n)
    return X, y


# ---------------------------------------------------------------------------
# Home-Credit-like DataFrames
# ---------------------------------------------------------------------------

_APPLICATION_COLUMNS = [
    "SK_ID_CURR",
    "TARGET",
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "CNT_FAM_MEMBERS",
    "NAME_HOUSING_TYPE",
]


@pytest.fixture()
def sample_application_df() -> pd.DataFrame:
    """100-row DataFrame mimicking ``application_train.csv``."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100_000, 100_000 + n),
            "TARGET": rng.binomial(1, 0.08, size=n),
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
            "CODE_GENDER": rng.choice(["M", "F"], n),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], n),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n),
            "CNT_CHILDREN": rng.integers(0, 4, size=n),
            "AMT_INCOME_TOTAL": rng.uniform(50_000, 500_000, n),
            "AMT_CREDIT": rng.uniform(100_000, 1_500_000, n),
            "AMT_ANNUITY": rng.uniform(5_000, 60_000, n),
            "AMT_GOODS_PRICE": rng.uniform(50_000, 1_200_000, n),
            "NAME_EDUCATION_TYPE": rng.choice(
                [
                    "Higher education",
                    "Secondary / secondary special",
                    "Incomplete higher",
                    "Lower secondary",
                    "Academic degree",
                ],
                n,
            ),
            "NAME_FAMILY_STATUS": rng.choice(["Married", "Single", "Separated"], n),
            "REGION_POPULATION_RELATIVE": rng.uniform(0.0, 0.1, n),
            "DAYS_BIRTH": -rng.integers(7_000, 25_000, size=n),
            "DAYS_EMPLOYED": -rng.integers(100, 15_000, size=n),
            "DAYS_REGISTRATION": -rng.uniform(500, 20_000, n),
            "DAYS_ID_PUBLISH": -rng.integers(100, 7_000, size=n),
            "EXT_SOURCE_1": rng.uniform(0, 1, n),
            "EXT_SOURCE_2": rng.uniform(0, 1, n),
            "EXT_SOURCE_3": rng.uniform(0, 1, n),
            "CNT_FAM_MEMBERS": rng.integers(1, 6, size=n).astype(float),
            "NAME_HOUSING_TYPE": rng.choice(
                ["House / apartment", "Rented apartment", "With parents"], n
            ),
        }
    )
    return df


@pytest.fixture()
def sample_features_df() -> pd.DataFrame:
    """200-row feature matrix with ``feat_*`` columns and id/target."""
    rng = np.random.default_rng(42)
    n = 200
    data: dict[str, np.ndarray] = {
        "SK_ID_CURR": np.arange(100_000, 100_000 + n),
        "TARGET": rng.binomial(1, 0.08, size=n),
    }
    feat_names = [
        "feat_income",
        "feat_log_income",
        "feat_credit_amount",
        "feat_annuity",
        "feat_goods_price",
        "feat_credit_to_goods",
        "feat_employed_flag",
        "feat_employment_years",
        "feat_age_years",
        "feat_age_bin",
        "feat_debt_burden",
        "feat_annuity_to_income",
        "feat_ext_source_1",
        "feat_ext_source_2",
        "feat_ext_source_3",
        "feat_bureau_total_count",
        "feat_bureau_active_ratio",
        "feat_cc_utilization_mean",
        "feat_trend_dpd_delta",
        "feat_education_ordinal",
    ]
    for name in feat_names:
        data[name] = rng.standard_normal(n)
    return pd.DataFrame(data)
