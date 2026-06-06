# Champion-Challenger Log

> Scope: this log documents the model comparisons that have been run under the champion-challenger governance framework. Each entry: who challenged, on what data, gate-by-gate result, and the promotion decision. Real measurements come from `artifacts/evaluation_results.json` and the MLflow runs under `mlruns/895473315040154611/`.

---

## 1. The governance gates

From `monitoring/champion_challenger.py`, a challenger can be promoted only if it passes **all four** gates:

| Gate | Threshold | Rationale |
|---|---|---|
| **Performance** | Challenger PR-AUC ≥ Champion PR-AUC | Challenger must not degrade the primary metric (PR-AUC at 8% prevalence). A tie (Δ=0) passes; any regression fails. |
| **Calibration** | Brier degradation ≤ 0.02 | Calibration drift of more than 2 Brier points invalidates the threshold policy. |
| **Fairness** | AIR within tolerance (≥ 0.80) and no regression in EO/DP gaps | A performance win that worsens fairness is not a promotion. |
| **Interpretability** | Challenger interpretability tier ≥ champion's (`low < medium < high`) | Must not trade explainability for marginal discrimination gains. |

If all four pass → `promote`. If performance/calibration fails outright → `keep_champion` (from `promote_decision`). If the subsequent fairness reconcile (`reconcile_fairness.py`) finds the champion itself fails a fairness gate, the disposition can be elevated to `investigate` even when the challenger already failed. If metrics are mixed without an outright performance/calibration failure → `investigate`.

---

## 2. LightGBM (champion) vs XGBoost (challenger)

### Setup

| Property | Value |
|---|---|
| MLflow champion run | `0bbae0e83d8447a089b709484d9c19ca` |
| MLflow challenger run | `b0ff3f4b78e54c10a8d36fa508c7be54` |
| Evaluation protocol | 5-fold StratifiedGroupKFold (grouped on `SK_ID_CURR`), OOF scores across 307,511 applicants |
| HPO | Optuna, 50 trials, PR-AUC objective (each model independently tuned on the same CV) |
| Feature set | Identical — 15 features from Boruta+VIF hybrid (source of truth: `mlruns/.../artifacts/model_metadata.json`) |
| Cost matrix | FP=$600, FN=$6000 (10:1 FN:FP ratio; see `decisioning/COST_MATRIX.md`) |

### Champion HPs (LightGBM)

| Param | Value |
|---|---|
| n_estimators | 717 |
| learning_rate | 0.0231 |
| max_depth | 7 |
| num_leaves | 42 |
| min_child_samples | 196 |
| scale_pos_weight | 10.41 |

### Challenger HPs (XGBoost)

| Param | Value |
|---|---|
| n_estimators | 1054 |
| learning_rate | 0.0741 |
| max_depth | 3 |
| min_child_weight | 47 |
| gamma | 2.32 |
| scale_pos_weight | 12.86 |

### Gate-by-gate result

From `artifacts/evaluation_results.json`:

| Gate | Champion | Challenger | Delta | Pass? |
|---|---|---|---|---|
| **Performance** — PR-AUC | 0.2467 | 0.2440 | **-0.0026** | **FAIL** (challenger worse, not allowed) |
| Performance — ROC-AUC (secondary) | 0.7621 | 0.7612 | -0.0009 | n/a (not gated, informational) |
| **Calibration** — Brier raw | 0.1773 | 0.2099 | **+0.0326** | **FAIL** (> 0.02 tolerance) |
| Calibration — Brier (Platt) | 0.0678 | 0.0679 | +0.0001 | Informational — both recover post-calibration |
| **Fairness** — AIR (gender, t=0.40) | 0.81 | n/a | — | PASS narrowly (≥0.80 floor; `governance/FAIRNESS_REPORT.md`) |
| **Fairness** — AIR (education, t=0.40) | 0.61 | n/a | — | **FAIL** 80% rule on education subgroup |
| **Interpretability** — tier | medium | medium | equal | PASS |

Measured AIR on the champion's OOF predictions at the `t=0.40` production decline gate: **gender 0.8125** (narrow pass) and **education 0.6056** (fails 80% rule). The challenger-side fairness numbers were not separately recomputed because the challenger failed the performance and calibration gates first and was never a live promotion candidate. See `governance/FAIRNESS_REPORT.md` section 1 for the full threshold sweep on the champion.

### Promotion decision — stage 1 (`champion_challenger.promote_decision`)

```
decision = "keep_champion"
notes = [
  "Challenger PR-AUC did not improve (-0.0026)",
  "Calibration degraded beyond tolerance: delta=+0.0326 > 0.02",
]
```

The first two notes block promotion of the challenger on their own (performance and calibration gates). `promote_decision` returns `"keep_champion"` when the performance gate fails.

### Promotion decision — stage 2 (`reconcile_fairness` override)

After `scripts/reconcile_fairness.py` recomputed multi-slice AIR from the champion's OOF predictions, it found `NAME_EDUCATION_TYPE` AIR = 0.6056 (fails 80% rule) and patched `artifacts/evaluation_results.json`:

```
decision = "investigate"   # overridden from "keep_champion" by reconcile_fairness.py
notes = [
  "Challenger PR-AUC did not improve (-0.0026)",
  "Calibration degraded beyond tolerance: delta=+0.0326 > 0.02",
  "Multi-slice AIR floor at t=0.4: 0.6056 on NAME_EDUCATION_TYPE (FAILS 80% rule).",
]
```

The final disposition written to `artifacts/evaluation_results.json` is `investigate` because neither model is a clean ship in the measured state: the challenger fails performance/calibration outright, and the champion fails fairness on the education subgroup. The third note is the fairness measurement on the **champion itself** — surfaced by the reconcile step against `artifacts/oof_predictions_scored.parquet`.

**What this means operationally.** The challenger does not promote — that part is unambiguous. The champion stays in the champion slot because there is no better candidate, but it is **not production-ready** as measured. Mitigation (reject-option classification at the review-band post-processing step; see `governance/FAIRNESS_REPORT.md` §1 "Mitigation considered") is documented and required before any deployment that would expose the model to real applicants.

This is a **failed promotion** on record — evidence the governance gate is not purely decorative. The calibration gate did real work: both models converge post-calibration, but the raw calibration gap is what the gate measures, because threshold policy is set on raw model output before Platt is applied.

### Downstream impact on decisioning

Even though the challenger has a slightly higher cost-optimal threshold (0.54 vs 0.50), at the **fixed production gates** (approve < 0.15, decline ≥ 0.40), the challenger:

- Auto-approves fewer applicants: 10.0% vs champion's 14.0%
- Sends fewer applicants to manual review: 38.8% vs champion's 43.8%, while auto-declining more: 51.2% vs champion's 42.2%

The challenger is more cautious — **not because it is more accurate**, but because its raw probabilities are pushed more aggressively toward 1. Operationally this would shrink the underwriter queue by ~11% while shifting more volume to automatic declines — with no discrimination gain. Another reason to keep the champion.

---

## 3. Planned comparisons

Documented methodology for additional comparisons that would strengthen the log. None have been executed yet.

### CatBoost as challenger (methodology)

CatBoost is the obvious next challenger because:

- Handles categorical features natively (no ordinal encoding of `NAME_EDUCATION_TYPE` needed).
- Typically better out-of-the-box calibration than XGBoost/LightGBM (ordered boosting).
- Different loss surface may move the 15-feature selector result.

Procedure:

1. Mirror `training/train.py`'s Optuna loop with `catboost.CatBoostClassifier`.
2. Run 50 trials on the same CV folds and same Boruta+VIF feature set.
3. `python -m scripts.evaluate_and_compare --challenger catboost` (needs a `--challenger` flag, currently the script hard-codes XGBoost — a small refactor).
4. Run the 4-gate comparison.

### Monotonic GBM as challenger (methodology)

For regulatory defensibility, some regimes prefer models where feature-target relationships are provably monotone (e.g. income↑ ⇒ PD↓). Both LightGBM and XGBoost support monotonic constraints.

Procedure:

1. Define monotonic constraints per feature in the domain dictionary (e.g. `feat_debt_burden` monotone increasing, `feat_employment_years` monotone decreasing).
2. Retrain champion with constraints. Expect ROC-AUC to drop modestly (~0.005-0.01) in exchange for full monotone compliance.
3. Run the 4-gate comparison with one extra gate: monotonicity PASS/FAIL via `features/quality_checks.py::monotonicity_check`.

### WoE scorecard as challenger (methodology)

The WoE scorecard is the interpretable baseline — usually deployed *alongside* the booster, but can also challenge for the champion slot if regulator pressure raises the interpretability gate.

Procedure:

1. Train scorecard on the same 15 features (or a scorecard-appropriate expansion) via `decisioning/scorecard.py`.
2. Score OOF on the same CV folds.
3. Run the 4-gate comparison. Expected outcome: scorecard loses the performance gate (~0.04-0.05 PR-AUC drop) but wins the interpretability gate (`high` > `medium`).
4. Decision: scorecard would not replace LightGBM for auto-decisioning, but would be formally registered as the **champion for the interpretability-constrained segment** (e.g. regulator-facing explanations, adverse-action review, and a governance dual-model check on feature driver alignment).

---

## 4. What a "passed" promotion would look like

For reference, a challenger that *would* pass the four gates would show:

| Gate | Example passing delta |
|---|---|
| Performance | PR-AUC +0.01 or better (clearly above the PR-AUC CV stddev of ~0.007 on the champion) |
| Calibration | Brier degradation < 0.01 |
| Fairness | No AIR drop below 0.80; no worsening of DP or EO gaps |
| Interpretability | Same tier or higher |

And the governance decision would read `promote` with no red flags in the notes. We do not yet have such a record in this log.

---

## 5. Rollback criteria

From `monitoring/champion_challenger.py::rollback_check()`:

- Rolling ROC-AUC drops > 3 pp from baseline `baseline_roc_auc: 0.762`.
- Rolling Brier increases > 5 pp from baseline `baseline_brier: 0.068`.

These are *monitoring* triggers on the deployed champion — independent of any challenger comparison. If rollback fires, the model is pulled and the last-known-good artefact is restored while triage begins.

---

## 6. Linked artefacts

- `artifacts/evaluation_results.json` — source of the comparison numbers.
- `mlruns/895473315040154611/` — raw MLflow runs for champion and challenger.
- `monitoring/champion_challenger.py` — `compare_models`, `promote_decision`, `rollback_check`.
- `scripts/evaluate_and_compare.py` — the harness that produces `evaluation_results.json`.
