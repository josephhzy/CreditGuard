"""Reconcile ``artifacts/evaluation_results.json`` with the measured AIRs.

Background. ``governance/FAIRNESS_REPORT.md`` measures CODE_GENDER AIR=0.8125
and NAME_EDUCATION_TYPE AIR=0.6056 at the production decline gate (t=0.40),
i.e. education fails the 80% rule. The pipeline-level
``evaluation_results.json`` was reporting AIR=1.0 and
``fairness_within_tolerance: true`` because ``compute_fairness_report`` was
only being invoked on a single sensitive column and a stale code path was
defaulting to a perfect-parity sentinel. Two governance artifacts cannot
disagree — pick the worst measured AIR across the slices we know about and
make ``fairness_within_tolerance`` reflect it.

This script:
  1. Recomputes per-slice AIRs from the OOF predictions parquet.
  2. Patches ``evaluation_results.json`` so the headline AIR is the
     worst-case across the slices, the per-slice numbers are written out,
     ``fairness_within_tolerance`` is recomputed, and the source of each
     measurement is recorded.

Usage:
    python -m scripts.reconcile_fairness
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import structlog

from evaluation.fairness import adverse_impact_ratio

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer(colors=True)],
    wrapper_class=structlog.make_filtering_bound_logger(0),
)
log = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OOF_PATH = PROJECT_ROOT / "artifacts" / "oof_predictions_scored.parquet"
EVAL_PATH = PROJECT_ROOT / "artifacts" / "evaluation_results.json"

DECLINE_GATE = 0.40
APPROVE_GATE = 0.15
FAIRNESS_TOLERANCE = 0.80


def _air_for_slice(
    df: pd.DataFrame,
    sensitive_col: str,
    threshold: float,
    drop_values: tuple[str, ...] = (),
) -> dict[str, float | int | bool | str]:
    """Compute AIR at ``threshold`` ignoring rows whose group is in ``drop_values``."""
    sub = df[~df[sensitive_col].isin(drop_values) & df[sensitive_col].notna()]
    y_pred = (sub["y_score"].to_numpy() >= threshold).astype(int)
    air = float(adverse_impact_ratio(y_pred, sub[sensitive_col]))
    return {
        "sensitive_attribute": sensitive_col,
        "threshold": threshold,
        "n_rows": int(len(sub)),
        "air": round(air, 4),
        "passes_80pct_rule": bool(air >= FAIRNESS_TOLERANCE),
    }


def main() -> None:
    if not OOF_PATH.exists():
        raise FileNotFoundError(
            f"OOF predictions not found at {OOF_PATH}. Run scripts/evaluate_and_compare.py first."
        )
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"evaluation_results.json not found at {EVAL_PATH}")

    df = pd.read_parquet(OOF_PATH)
    log.info("oof_loaded", n_rows=len(df))

    slices = [
        # XNA is a Home Credit sentinel for "missing", not a third group.
        _air_for_slice(df, "CODE_GENDER", DECLINE_GATE, drop_values=("XNA",)),
        _air_for_slice(df, "CODE_GENDER", APPROVE_GATE, drop_values=("XNA",)),
        _air_for_slice(df, "NAME_EDUCATION_TYPE", DECLINE_GATE),
        _air_for_slice(df, "NAME_EDUCATION_TYPE", APPROVE_GATE),
    ]

    # Headline AIR = worst across slices at the decline gate (the operating point
    # that maps to the DECLINE decision band in the served decisioning).
    decline_slices = [s for s in slices if s["threshold"] == DECLINE_GATE]
    worst_decline = min(decline_slices, key=lambda s: float(s["air"]))
    headline_air = float(worst_decline["air"])
    fairness_ok = bool(headline_air >= FAIRNESS_TOLERANCE)

    log.info(
        "headline_air",
        air=headline_air,
        sensitive=worst_decline["sensitive_attribute"],
        passes_80pct_rule=fairness_ok,
    )

    with open(EVAL_PATH) as f:
        eval_results = json.load(f)

    eval_results["champion"]["metrics"]["adverse_impact_ratio"] = headline_air

    # Mark challenger fairness fields as not reconciled.
    # The OOF parquet stores champion scores only; challenger AIR cannot be
    # recomputed here. The stale sentinel values (1.0 / 0.0 / 0.0) are replaced
    # with None (JSON null) so governance readers never treat "unmeasured" as "passing".
    if "challenger" in eval_results and "metrics" in eval_results["challenger"]:
        eval_results["challenger"]["metrics"]["adverse_impact_ratio"] = None
        eval_results["challenger"]["metrics"]["demographic_parity_diff"] = None
        eval_results["challenger"]["metrics"]["equalized_odds_diff"] = None
        eval_results["challenger"]["metrics"]["fairness_note"] = (
            "Fairness fields not reconciled: challenger OOF scores are not "
            "persisted. See fairness_slices (champion-derived) for the "
            "system-level AIR assessment."
        )

    eval_results["fairness_slices"] = {
        "tolerance": FAIRNESS_TOLERANCE,
        "headline_slice": worst_decline["sensitive_attribute"],
        "headline_threshold": DECLINE_GATE,
        "headline_air": headline_air,
        "passes_80pct_rule": fairness_ok,
        "per_slice": slices,
        "source": "scripts/reconcile_fairness.py against artifacts/oof_predictions_scored.parquet",
        "see_also": "governance/FAIRNESS_REPORT.md section 1",
    }
    eval_results["comparison"]["fairness_within_tolerance"] = fairness_ok

    notes: list[str] = list(eval_results["comparison"].get("notes", []))
    fairness_note = (
        f"Multi-slice AIR floor at t={DECLINE_GATE}: "
        f"{headline_air:.4f} on {worst_decline['sensitive_attribute']} "
        f"({'passes' if fairness_ok else 'FAILS'} 80% rule)."
    )
    notes = [n for n in notes if "AIR" not in n and "Fairness" not in n]
    notes.append(fairness_note)
    eval_results["comparison"]["notes"] = notes

    if not fairness_ok:
        eval_results["comparison"]["decision"] = "investigate"

    with open(EVAL_PATH, "w") as f:
        json.dump(eval_results, f, indent=2)
    log.info("evaluation_results_patched", path=str(EVAL_PATH))


if __name__ == "__main__":
    main()
