"""Feature selection methods for credit scoring models.

Provides Boruta, Recursive Feature Elimination, Variance Inflation
Factor filtering, and a hybrid pipeline that combines all three to
select the most predictive, non-redundant feature set.

Typical usage::

    from training.feature_selection import hybrid_select

    selected, report = hybrid_select(X_train, y_train)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import pandas as pd
import structlog
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SelectionReport:
    """Report produced by each feature selection method."""

    method: str
    n_input_features: int
    n_selected_features: int
    selected_features: list[str]
    dropped_features: list[str]
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Boruta
# ---------------------------------------------------------------------------


def boruta_select(
    X: pd.DataFrame,
    y: pd.Series,
    max_iter: int = 200,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[list[str], SelectionReport]:
    """Select features using the Boruta algorithm.

    Boruta wraps a random forest and iteratively tests whether each
    feature is significantly more important than shuffled shadow copies.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Binary target.
    max_iter:
        Maximum Boruta iterations.
    alpha:
        Significance level for feature acceptance.
    random_state:
        Random seed.

    Returns
    -------
    tuple[list[str], SelectionReport]
        Selected feature names and a detailed report.
    """
    logger.info("boruta_start", n_features=X.shape[1], max_iter=max_iter, alpha=alpha)

    rf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=random_state,
        max_depth=7,
    )
    selector = BorutaPy(
        estimator=rf,
        n_estimators="auto",
        max_iter=max_iter,
        alpha=alpha,
        random_state=random_state,
        verbose=0,
    )
    selector.fit(X.values, y.values)

    selected_mask = selector.support_
    selected = X.columns[selected_mask].tolist()
    tentative = X.columns[selector.support_weak_].tolist()
    dropped = X.columns[~selected_mask].tolist()

    report = SelectionReport(
        method="boruta",
        n_input_features=X.shape[1],
        n_selected_features=len(selected),
        selected_features=selected,
        dropped_features=dropped,
        details={
            "ranking": dict(zip(X.columns.tolist(), selector.ranking_.tolist(), strict=False)),
            "tentative_features": tentative,
            "n_confirmed_features": selector.n_features_,
        },
    )
    logger.info(
        "boruta_done",
        n_selected=len(selected),
        n_tentative=len(tentative),
        n_dropped=len(dropped),
    )
    return selected, report


# ---------------------------------------------------------------------------
# Recursive Feature Elimination
# ---------------------------------------------------------------------------


def rfe_select(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 50,
    step: int = 5,
    random_state: int = 42,
) -> tuple[list[str], SelectionReport]:
    """Select features via Recursive Feature Elimination (RFE).

    Uses a random forest estimator and eliminates ``step`` features per
    iteration until ``n_features`` remain.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Binary target.
    n_features:
        Target number of features.
    step:
        Number of features to remove per iteration.
    random_state:
        Random seed.

    Returns
    -------
    tuple[list[str], SelectionReport]
        Selected feature names and a detailed report.
    """
    logger.info("rfe_start", n_features_target=n_features, step=step)

    rf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=random_state,
        max_depth=7,
    )
    rfe = RFE(estimator=rf, n_features_to_select=n_features, step=step)
    rfe.fit(X, y)

    selected_mask = rfe.support_
    selected = X.columns[selected_mask].tolist()
    dropped = X.columns[~selected_mask].tolist()

    report = SelectionReport(
        method="rfe",
        n_input_features=X.shape[1],
        n_selected_features=len(selected),
        selected_features=selected,
        dropped_features=dropped,
        details={
            "ranking": dict(zip(X.columns.tolist(), rfe.ranking_.tolist(), strict=False)),
            "n_features_target": n_features,
            "step": step,
        },
    )
    logger.info("rfe_done", n_selected=len(selected))
    return selected, report


# ---------------------------------------------------------------------------
# Variance Inflation Factor
# ---------------------------------------------------------------------------


def vif_filter(
    X: pd.DataFrame,
    threshold: float = 10.0,
) -> tuple[list[str], SelectionReport]:
    """Remove features with Variance Inflation Factor above threshold.

    VIF measures multicollinearity.  Features are dropped iteratively
    (highest VIF first) until all remaining features have VIF below the
    threshold.

    Parameters
    ----------
    X:
        Feature matrix (numeric only).
    threshold:
        Maximum acceptable VIF.

    Returns
    -------
    tuple[list[str], SelectionReport]
        Surviving feature names and a detailed report.
    """
    logger.info("vif_filter_start", n_features=X.shape[1], threshold=threshold)

    # Work on a copy; fill any NaN with column median for VIF computation
    Xc = X.copy()
    Xc = Xc.fillna(Xc.median())

    dropped: list[str] = []
    vif_history: list[dict[str, float]] = []

    while True:
        cols = Xc.columns.tolist()
        if len(cols) <= 1:
            break

        vif_values = np.array([variance_inflation_factor(Xc.values, i) for i in range(len(cols))])
        vif_dict = dict(zip(cols, vif_values.tolist(), strict=False))
        vif_history.append(vif_dict)

        max_vif_idx = int(np.argmax(vif_values))
        max_vif = float(vif_values[max_vif_idx])

        if max_vif <= threshold:
            break

        drop_col = cols[max_vif_idx]
        logger.debug("vif_drop", feature=drop_col, vif=round(max_vif, 2))
        dropped.append(drop_col)
        Xc = Xc.drop(columns=[drop_col])

    selected = Xc.columns.tolist()

    report = SelectionReport(
        method="vif",
        n_input_features=X.shape[1],
        n_selected_features=len(selected),
        selected_features=selected,
        dropped_features=dropped,
        details={
            "threshold": threshold,
            "n_iterations": len(vif_history),
            "final_vif": vif_history[-1] if vif_history else {},
        },
    )
    logger.info("vif_filter_done", n_selected=len(selected), n_dropped=len(dropped))
    return selected, report


# ---------------------------------------------------------------------------
# Hybrid pipeline
# ---------------------------------------------------------------------------


def hybrid_select(
    X: pd.DataFrame,
    y: pd.Series,
    boruta_max_iter: int = 200,
    boruta_alpha: float = 0.05,
    vif_threshold: float = 10.0,
    random_state: int = 42,
) -> tuple[list[str], SelectionReport]:
    """Hybrid feature selection: Boruta -> VIF filter -> importance rank.

    1. Run Boruta to identify statistically relevant features.
    2. Apply VIF filter on the Boruta-selected set to remove
       multicollinear features.
    3. Rank surviving features by random forest importance.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Binary target.
    boruta_max_iter:
        Maximum Boruta iterations.
    boruta_alpha:
        Boruta significance level.
    vif_threshold:
        Maximum VIF for multicollinearity filtering.
    random_state:
        Random seed.

    Returns
    -------
    tuple[list[str], SelectionReport]
        Final selected feature names and a combined report.
    """
    logger.info("hybrid_select_start", n_features=X.shape[1])

    # Stage 1: Boruta
    boruta_features, boruta_report = boruta_select(
        X,
        y,
        max_iter=boruta_max_iter,
        alpha=boruta_alpha,
        random_state=random_state,
    )
    logger.info("hybrid_stage_1_boruta", n_surviving=len(boruta_features))

    if len(boruta_features) == 0:
        logger.warning("boruta_selected_zero_features_falling_back_to_all")
        boruta_features = X.columns.tolist()

    # Stage 2: VIF filter on Boruta survivors
    X_boruta = X[boruta_features]
    vif_features, vif_report = vif_filter(X_boruta, threshold=vif_threshold)
    logger.info("hybrid_stage_2_vif", n_surviving=len(vif_features))

    # Stage 3: Rank by RF importance
    X_final = X[vif_features]
    rf = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=random_state,
        max_depth=7,
    )
    rf.fit(X_final, y)
    importances = pd.Series(rf.feature_importances_, index=vif_features).sort_values(
        ascending=False
    )
    importance_ranking = importances.to_dict()

    selected = vif_features  # all VIF survivors are kept; importance is for ranking

    report = SelectionReport(
        method="hybrid",
        n_input_features=X.shape[1],
        n_selected_features=len(selected),
        selected_features=selected,
        dropped_features=sorted(set(X.columns) - set(selected)),
        details={
            "boruta_report": {
                "n_selected": boruta_report.n_selected_features,
                "tentative": boruta_report.details.get("tentative_features", []),
            },
            "vif_report": {
                "n_dropped": len(vif_report.dropped_features),
                "threshold": vif_threshold,
            },
            "importance_ranking": importance_ranking,
        },
    )
    logger.info("hybrid_select_done", n_final=len(selected))
    return selected, report


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: Literal["boruta", "rfe", "vif", "hybrid"] = "hybrid",
    **kwargs,
) -> tuple[list[str], SelectionReport]:
    """Dispatch to the requested feature selection method.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Binary target.
    method:
        Selection algorithm name.
    **kwargs:
        Forwarded to the underlying method.

    Returns
    -------
    tuple[list[str], SelectionReport]
        Selected feature names and report.
    """
    dispatch: dict[str, Callable[..., tuple[list[str], SelectionReport]]] = {
        "boruta": boruta_select,
        "rfe": rfe_select,
        "vif": lambda X, y, **kw: vif_filter(
            X, **{k: v for k, v in kw.items() if k == "threshold"}
        ),
        "hybrid": hybrid_select,
    }
    if method not in dispatch:
        raise ValueError(f"Unknown feature selection method: '{method}'")

    logger.info("feature_selection_dispatch", method=method)
    return dispatch[method](X, y, **kwargs)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import yaml

    parser = argparse.ArgumentParser(description="Run feature selection on processed features")
    parser.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    parser.add_argument("--method", type=str, default=None, help="Override selection method")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    features_path = Path(cfg["data"]["features_path"])
    target_col = cfg["data"]["target_column"]
    fs_cfg = cfg["feature_selection"]

    if not features_path.exists():
        logger.error("features_not_found", path=str(features_path))
        raise SystemExit(1)

    df = pd.read_parquet(features_path)
    method = args.method or fs_cfg["method"]

    X = df.drop(columns=[target_col, cfg["data"]["id_column"]], errors="ignore")
    y = df[target_col]

    selected, report = select_features(X, y, method=method)

    print(f"\nFeature Selection Report ({report.method})")
    print(f"  Input features:    {report.n_input_features}")
    print(f"  Selected features: {report.n_selected_features}")
    print(f"  Dropped features:  {len(report.dropped_features)}")
    print(f"\n  Top selected: {selected[:20]}")
