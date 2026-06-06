"""Join all Home Credit tables into a single modelling-ready DataFrame.

The Home Credit dataset consists of a star schema centred on
``application_train`` / ``application_test`` (keyed by ``SK_ID_CURR``).
Secondary tables contain one-to-many historical records that must be
aggregated to one row per applicant before joining back.

Relationship map::

    application_{train,test}  (SK_ID_CURR — base)
        |
        +-- bureau  (SK_ID_CURR -> many)
        |       +-- bureau_balance  (SK_ID_BUREAU -> many)
        |
        +-- previous_application  (SK_ID_CURR -> many)
                +-- POS_CASH_balance        (SK_ID_PREV -> many)
                +-- installments_payments   (SK_ID_PREV -> many)
                +-- credit_card_balance     (SK_ID_PREV -> many)

Each secondary table is aggregated using mean, max, min, count, and sum
for numeric columns, and mode / nunique for categoricals. The result is
saved as Parquet in the processed directory.

Typical usage::

    python -m data.join_tables                 # build from application_train
    python -m data.join_tables --test          # build from application_test
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "train.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

_NUMERIC_AGGS: list[str] = ["mean", "max", "min", "sum", "count"]


def _safe_mode(s: pd.Series) -> pd.Series:
    """Return the mode of a Series (first value if tied), or NaN if empty."""
    m = s.mode()
    return m.iloc[0] if len(m) > 0 else np.nan


def _aggregate_numeric(
    df: pd.DataFrame,
    group_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Aggregate numeric columns to one row per *group_col*.

    For each numeric column ``c``, produces:
    ``{prefix}_{c}_mean``, ``{prefix}_{c}_max``, ``{prefix}_{c}_min``,
    ``{prefix}_{c}_sum``, ``{prefix}_{c}_count``.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    # Remove the grouping column itself from aggregation
    num_cols = [c for c in num_cols if c != group_col]

    if not num_cols:
        # Nothing to aggregate — just return row count
        agg_df = df.groupby(group_col).size().to_frame(f"{prefix}_row_count")
        return agg_df.reset_index()

    agg_dict = {c: _NUMERIC_AGGS for c in num_cols}
    agg_df = df.groupby(group_col)[num_cols].agg(agg_dict)

    # Flatten multi-level column names
    agg_df.columns = [f"{prefix}_{col}_{stat}" for col, stat in agg_df.columns]
    agg_df = agg_df.reset_index()

    # Add total row count per group
    counts = df.groupby(group_col).size().to_frame(f"{prefix}_row_count")
    agg_df = agg_df.merge(counts, on=group_col, how="left")

    return agg_df


def _aggregate_categorical(
    df: pd.DataFrame,
    group_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Aggregate categorical columns: nunique and mode per group."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return pd.DataFrame({group_col: df[group_col].unique()})

    parts: list[pd.DataFrame] = []
    for col in cat_cols:
        nunique = df.groupby(group_col)[col].nunique().to_frame(f"{prefix}_{col}_nunique")
        mode = df.groupby(group_col)[col].agg(_safe_mode).to_frame(f"{prefix}_{col}_mode")
        part = nunique.join(mode)
        parts.append(part)

    result = pd.concat(parts, axis=1).reset_index()
    return result


def aggregate_table(
    df: pd.DataFrame,
    group_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Aggregate *df* to one row per *group_col* combining numeric and categorical features.

    Parameters
    ----------
    df:
        Raw secondary table.
    group_col:
        Column to group by (``SK_ID_CURR`` or ``SK_ID_PREV``).
    prefix:
        String prefix for generated column names (e.g. ``"bureau"``).

    Returns
    -------
    pd.DataFrame
        One row per unique value of *group_col*.
    """
    logger.info("aggregating_table", prefix=prefix, group_col=group_col, rows=len(df))

    num_agg = _aggregate_numeric(df, group_col, prefix)
    cat_agg = _aggregate_categorical(df, group_col, prefix)
    merged = num_agg.merge(cat_agg, on=group_col, how="outer")

    logger.info(
        "aggregation_complete",
        prefix=prefix,
        result_rows=len(merged),
        result_cols=len(merged.columns),
    )
    return merged


# ---------------------------------------------------------------------------
# Two-level aggregation helper (e.g. bureau_balance -> bureau -> application)
# ---------------------------------------------------------------------------


def _aggregate_two_level(
    child_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    child_key: str,
    parent_key: str,
    final_key: str,
    child_prefix: str,
    combined_prefix: str,
) -> pd.DataFrame:
    """Aggregate a child table through its parent to the final grouping key.

    For example, ``bureau_balance`` is grouped by ``SK_ID_BUREAU`` first,
    merged onto ``bureau`` to get ``SK_ID_CURR``, then grouped again by
    ``SK_ID_CURR``.
    """
    # First level: child -> parent key
    child_agg = aggregate_table(child_df, child_key, child_prefix)

    # Merge parent mapping (to get final_key for each child_key)
    mapping = parent_df[[child_key, final_key]].drop_duplicates()
    child_with_fk = child_agg.merge(mapping, on=child_key, how="left")

    # Second level: aggregate to final_key
    child_with_fk = child_with_fk.drop(columns=[child_key])
    result = aggregate_table(child_with_fk, final_key, combined_prefix)
    return result


# ---------------------------------------------------------------------------
# Main join pipeline
# ---------------------------------------------------------------------------


def _read_csv(raw_dir: Path, name: str, nrows: int | None = None) -> pd.DataFrame:
    """Read a CSV from the raw directory with optional row limit."""
    path = raw_dir / f"{name}.csv"
    if not path.exists():
        logger.warning("csv_not_found", file=str(path))
        return pd.DataFrame()
    logger.info("reading_csv", file=name, path=str(path))
    return pd.read_csv(path, nrows=nrows)


def build_joined_dataset(
    config_path: Path | None = None,
    *,
    use_test: bool = False,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Join all Home Credit tables into one modelling-ready DataFrame.

    Parameters
    ----------
    config_path:
        Path to ``config/train.yaml``.
    use_test:
        When ``True``, build from ``application_test`` instead of
        ``application_train``.
    nrows:
        Optional row limit per CSV (useful for development).

    Returns
    -------
    pd.DataFrame
        A single DataFrame with one row per ``SK_ID_CURR``.
    """
    cfg = _load_config(config_path)
    root = _project_root()
    raw_dir = root / cfg["data"]["raw_dir"]
    processed_dir = root / cfg["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    id_col: str = cfg["data"]["id_column"]  # SK_ID_CURR

    # ----- Base table -----
    base_name = "application_test" if use_test else "application_train"
    base = _read_csv(raw_dir, base_name, nrows=nrows)
    if base.empty:
        raise FileNotFoundError(f"{base_name}.csv not found in {raw_dir}")
    logger.info("base_table_loaded", name=base_name, shape=base.shape)

    # ----- Bureau -----
    bureau = _read_csv(raw_dir, "bureau", nrows=nrows)
    if not bureau.empty:
        bureau_agg = aggregate_table(bureau, id_col, "BUREAU")
        base = base.merge(bureau_agg, on=id_col, how="left")
        logger.info("merged_bureau", shape=base.shape)

        # Bureau balance (two-level: SK_ID_BUREAU -> SK_ID_CURR)
        bb = _read_csv(raw_dir, "bureau_balance", nrows=nrows)
        if not bb.empty:
            bb_agg = _aggregate_two_level(
                child_df=bb,
                parent_df=bureau,
                child_key="SK_ID_BUREAU",
                parent_key="SK_ID_BUREAU",
                final_key=id_col,
                child_prefix="BB",
                combined_prefix="BB",
            )
            base = base.merge(bb_agg, on=id_col, how="left")
            logger.info("merged_bureau_balance", shape=base.shape)

    # ----- Previous application -----
    prev = _read_csv(raw_dir, "previous_application", nrows=nrows)
    if not prev.empty:
        prev_agg = aggregate_table(prev, id_col, "PREV")
        base = base.merge(prev_agg, on=id_col, how="left")
        logger.info("merged_previous_application", shape=base.shape)

        # POS_CASH_balance (two-level: SK_ID_PREV -> SK_ID_CURR)
        pos = _read_csv(raw_dir, "POS_CASH_balance", nrows=nrows)
        if not pos.empty:
            pos_agg = _aggregate_two_level(
                child_df=pos,
                parent_df=prev,
                child_key="SK_ID_PREV",
                parent_key="SK_ID_PREV",
                final_key=id_col,
                child_prefix="POS",
                combined_prefix="POS",
            )
            base = base.merge(pos_agg, on=id_col, how="left")
            logger.info("merged_pos_cash", shape=base.shape)

        # Installments payments (two-level: SK_ID_PREV -> SK_ID_CURR)
        inst = _read_csv(raw_dir, "installments_payments", nrows=nrows)
        if not inst.empty:
            inst_agg = _aggregate_two_level(
                child_df=inst,
                parent_df=prev,
                child_key="SK_ID_PREV",
                parent_key="SK_ID_PREV",
                final_key=id_col,
                child_prefix="INSTAL",
                combined_prefix="INSTAL",
            )
            base = base.merge(inst_agg, on=id_col, how="left")
            logger.info("merged_installments", shape=base.shape)

        # Credit card balance (two-level: SK_ID_PREV -> SK_ID_CURR)
        cc = _read_csv(raw_dir, "credit_card_balance", nrows=nrows)
        if not cc.empty:
            cc_agg = _aggregate_two_level(
                child_df=cc,
                parent_df=prev,
                child_key="SK_ID_PREV",
                parent_key="SK_ID_PREV",
                final_key=id_col,
                child_prefix="CC",
                combined_prefix="CC",
            )
            base = base.merge(cc_agg, on=id_col, how="left")
            logger.info("merged_credit_card", shape=base.shape)

    # ----- Save -----
    # Downstream consumers (`features/build_all.py`,
    # `scripts/evaluate_and_compare.py`) read `joined.parquet` for the
    # training path. Keep the `_test` suffix for the test path because that
    # output flows into the inference/holdout pipeline, which is distinct.
    out_name = "joined_test.parquet" if use_test else "joined.parquet"
    out_path = processed_dir / out_name
    base.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("saved_joined_dataset", path=str(out_path), shape=base.shape)

    return base


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Join Home Credit tables into one dataset")
    parser.add_argument("--test", action="store_true", help="Build from application_test")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows per CSV (dev mode)")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    args = parser.parse_args()

    df = build_joined_dataset(config_path=args.config, use_test=args.test, nrows=args.nrows)
    print(f"Joined dataset: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
