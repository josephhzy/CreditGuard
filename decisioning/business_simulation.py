"""Business impact simulation for credit decisioning thresholds.

Simulates the operational impact of different approval/review/decline
thresholds, including approval rates, expected default rates among
approved applicants, loss reduction compared to a no-model baseline,
and review queue sizing.

Enables business stakeholders to explore the cost-accuracy trade-off
before deploying threshold changes to production.

Typical usage::

    from decisioning.business_simulation import simulate, sweep_thresholds

    result = simulate(y_true, y_score, thresholds)
    sweep_df = sweep_thresholds(y_true, y_score, n_points=50)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
import yaml

logger = structlog.get_logger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "train.yaml"


def _load_cost_matrix(config_path: Path | None = None) -> dict[str, float]:
    """Load the cost matrix from the project config."""
    path = config_path or _CONFIG_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg["evaluation"]["threshold"]["cost_matrix"]


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationResult:
    """Outcome of a threshold simulation run."""

    approve_upper: float
    decline_lower: float
    approval_rate: float
    decline_rate: float
    review_rate: float
    expected_default_rate_approved: float
    expected_loss_reduction: float
    review_queue_size: int
    total_expected_cost: float
    n_approved: int
    n_declined: int
    n_review: int
    n_total: int

    def summary(self) -> dict[str, Any]:
        """Return a flat dict for DataFrame construction."""
        return {
            "approve_upper": self.approve_upper,
            "decline_lower": self.decline_lower,
            "approval_rate": round(self.approval_rate, 4),
            "decline_rate": round(self.decline_rate, 4),
            "review_rate": round(self.review_rate, 4),
            "expected_default_rate_approved": round(self.expected_default_rate_approved, 4),
            "expected_loss_reduction": round(self.expected_loss_reduction, 4),
            "review_queue_size": self.review_queue_size,
            "total_expected_cost": round(self.total_expected_cost, 2),
            "n_approved": self.n_approved,
            "n_declined": self.n_declined,
            "n_review": self.n_review,
            "n_total": self.n_total,
        }


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    approve_upper: float,
    decline_lower: float,
    *,
    config_path: Path | None = None,
) -> SimulationResult:
    """Simulate the operational impact of a given set of thresholds.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (1 = default, 0 = non-default).
    y_score:
        Raw model risk scores in [0, 1] (raw-score scale used by the
        decisioning bands, not the calibrated PD).
    approve_upper:
        Scores below this are approved.
    decline_lower:
        Scores at or above this are declined. In-between goes to review.
    config_path:
        Optional path to config YAML for cost matrix.

    Returns
    -------
    SimulationResult
        Full simulation metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")
    if y_true.shape[0] == 0:
        raise ValueError("Cannot simulate on empty data.")

    n_total = len(y_true)
    costs = _load_cost_matrix(config_path)
    # Defaults mirror config/train.yaml (Basel-derived book averages, 10:1
    # ratio). See decisioning/COST_MATRIX.md for the derivation.
    fp_cost = costs.get("fp_cost", 600)
    fn_cost = costs.get("fn_cost", 6000)

    # Assign bands
    approved_mask = y_score < approve_upper
    declined_mask = y_score >= decline_lower
    review_mask = ~approved_mask & ~declined_mask

    n_approved = int(approved_mask.sum())
    n_declined = int(declined_mask.sum())
    n_review = int(review_mask.sum())

    # Default rate among approved
    default_rate_approved = float(y_true[approved_mask].mean()) if n_approved > 0 else 0.0

    # Loss reduction vs. no-model baseline (approve everyone)
    baseline_default_rate = float(y_true.mean()) if n_total > 0 else 0.0
    if baseline_default_rate > 0:
        loss_reduction = (
            1.0 - (default_rate_approved * (n_approved / n_total)) / baseline_default_rate
        )
    else:
        loss_reduction = 0.0

    # Total expected cost:
    # FP cost = good applicants declined (lost business)
    # FN cost = bad applicants approved (default losses)
    n_fp = int((~y_true.astype(bool) & declined_mask).sum())  # good applicants declined
    n_fn = int((y_true.astype(bool) & approved_mask).sum())  # defaulters approved
    total_cost = n_fp * fp_cost + n_fn * fn_cost

    result = SimulationResult(
        approve_upper=approve_upper,
        decline_lower=decline_lower,
        approval_rate=n_approved / n_total,
        decline_rate=n_declined / n_total,
        review_rate=n_review / n_total,
        expected_default_rate_approved=default_rate_approved,
        expected_loss_reduction=loss_reduction,
        review_queue_size=n_review,
        total_expected_cost=total_cost,
        n_approved=n_approved,
        n_declined=n_declined,
        n_review=n_review,
        n_total=n_total,
    )

    logger.info(
        "simulation_complete",
        approve_upper=approve_upper,
        decline_lower=decline_lower,
        approval_rate=round(result.approval_rate, 4),
        default_rate_approved=round(result.expected_default_rate_approved, 4),
        total_cost=round(total_cost, 2),
    )
    return result


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


def sweep_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_points: int = 50,
    config_path: Path | None = None,
) -> pd.DataFrame:
    """Sweep across threshold pairs and compute metrics for each.

    Generates a grid of (approve_upper, decline_lower) combinations
    where approve_upper < decline_lower, and runs a simulation for each.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_score:
        Raw model risk scores.
    n_points:
        Number of candidate threshold values to sample along [0, 1].
    config_path:
        Optional config path for cost matrix.

    Returns
    -------
    pd.DataFrame
        One row per valid threshold pair with all simulation metrics.
    """
    candidates = np.linspace(0.01, 0.99, n_points)
    results: list[dict[str, Any]] = []

    for approve_upper in candidates:
        for decline_lower in candidates:
            if decline_lower <= approve_upper:
                continue
            try:
                sim = simulate(
                    y_true,
                    y_score,
                    approve_upper=float(approve_upper),
                    decline_lower=float(decline_lower),
                    config_path=config_path,
                )
                results.append(sim.summary())
            except ValueError:
                continue

    df = pd.DataFrame(results)
    logger.info("threshold_sweep_complete", n_combinations=len(df))
    return df


# ---------------------------------------------------------------------------
# Optimal operating point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimalPoint:
    """Result of optimal threshold search."""

    approve_upper: float
    decline_lower: float
    simulation: SimulationResult
    constraint_satisfied: bool
    note: str


def find_optimal_operating_point(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    max_default_rate: float = 0.05,
    min_approval_rate: float = 0.0,
    n_points: int = 50,
    config_path: Path | None = None,
) -> OptimalPoint:
    """Find the threshold pair that minimises total cost subject to constraints.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_score:
        Raw model risk scores.
    max_default_rate:
        Maximum acceptable default rate among approved applicants.
    min_approval_rate:
        Minimum required approval rate.
    n_points:
        Resolution of the threshold grid.
    config_path:
        Optional config path.

    Returns
    -------
    OptimalPoint
        The best threshold pair and its simulation result.
    """
    sweep_df = sweep_thresholds(y_true, y_score, n_points=n_points, config_path=config_path)

    if sweep_df.empty:
        raise ValueError("No valid threshold combinations found in sweep.")

    # Filter to feasible region
    feasible = sweep_df[
        (sweep_df["expected_default_rate_approved"] <= max_default_rate)
        & (sweep_df["approval_rate"] >= min_approval_rate)
    ]

    if feasible.empty:
        # Relax: pick the one closest to the constraint
        logger.warning(
            "no_feasible_point",
            max_default_rate=max_default_rate,
            min_approval_rate=min_approval_rate,
        )
        best_idx = sweep_df["total_expected_cost"].idxmin()
        row = sweep_df.loc[best_idx]
        sim = simulate(
            y_true,
            y_score,
            approve_upper=row["approve_upper"],
            decline_lower=row["decline_lower"],
            config_path=config_path,
        )
        return OptimalPoint(
            approve_upper=row["approve_upper"],
            decline_lower=row["decline_lower"],
            simulation=sim,
            constraint_satisfied=False,
            note="No combination satisfied all constraints. Returned lowest-cost point.",
        )

    best_idx = feasible["total_expected_cost"].idxmin()
    row = feasible.loc[best_idx]
    sim = simulate(
        y_true,
        y_score,
        approve_upper=row["approve_upper"],
        decline_lower=row["decline_lower"],
        config_path=config_path,
    )

    logger.info(
        "optimal_point_found",
        approve_upper=row["approve_upper"],
        decline_lower=row["decline_lower"],
        cost=row["total_expected_cost"],
    )

    return OptimalPoint(
        approve_upper=row["approve_upper"],
        decline_lower=row["decline_lower"],
        simulation=sim,
        constraint_satisfied=True,
        note=f"Optimal within constraints: default_rate<={max_default_rate}, approval_rate>={min_approval_rate}.",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Demo with synthetic data
    rng = np.random.default_rng(42)
    n = 5000
    y_true_demo = rng.binomial(1, 0.08, size=n)
    y_score_demo = np.clip(
        rng.beta(2, 10, size=n) + y_true_demo * rng.uniform(0.1, 0.3, size=n),
        0,
        1,
    )

    print("=== Single simulation ===")
    result = simulate(y_true_demo, y_score_demo, approve_upper=0.15, decline_lower=0.40)
    for k, v in result.summary().items():
        print(f"  {k}: {v}")

    print("\n=== Optimal operating point ===")
    optimal = find_optimal_operating_point(
        y_true_demo, y_score_demo, max_default_rate=0.04, n_points=20
    )
    print(
        f"  Thresholds: approve<{optimal.approve_upper:.3f}, decline>={optimal.decline_lower:.3f}"
    )
    print(f"  Constraint satisfied: {optimal.constraint_satisfied}")
    print(f"  Note: {optimal.note}")
    for k, v in optimal.simulation.summary().items():
        print(f"  {k}: {v}")
