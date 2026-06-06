# Retrain Policy

> Scope: concrete thresholds, sustained-alert logic, and the retrain workflow that CreditGuard follows when drift or performance monitors fire.

---

## 1. The two triggers

A retrain is triggered when **either**:

1. **Sustained drift alert.** `PSI > 0.20` (red) on any feature, or on the score distribution, sustained for **7 consecutive days**. Short-lived amber alerts (below 7 days) file a ticket but do not trigger retrain.
2. **Performance degradation.** Rolling 30-day ROC-AUC drops below **0.72** (vs. the 0.762 baseline), OR rolling PR-AUC drops more than **3 percentage points** below baseline (0.247 → 0.217), OR rolling Brier score increases by more than 0.05 (vs. the calibrated baseline of 0.068).

The **OR** is deliberate: drift can be a leading indicator of performance decay, and performance decay can happen without obvious feature drift (concept drift). We want either trigger to fire, not require both.

---

## 2. Exact thresholds

From `config/train.yaml::monitoring`:

| Trigger | Metric | Threshold | Window |
|---|---|---|---|
| Drift | PSI (feature or score) | ≥ 0.20 (red) | Sustained 7 days |
| Drift | PSI (feature or score) | ≥ 0.10, < 0.20 (amber) | Sustained 14 days |
| Performance | Rolling ROC-AUC | < 0.72 | 30 days |
| Performance | Rolling PR-AUC | Below baseline − 0.03 = 0.217 | 30 days |
| Performance | Rolling Brier | Above calibrated baseline + 0.05 = 0.118 | 30 days |
| Calibration | Calibration gap (by decile) | > 0.10 on any two deciles | 30 days |
| Fairness | AIR | < 0.80 | Any single measurement on monthly fairness review |
| Fairness | EO / DP gap | Increases by > 0.05 from baseline | Any single measurement |

---

## 3. Why 0.72 and 0.217?

The baseline numbers come from `artifacts/evaluation_results.json` — OOF on the training-era data:

- ROC-AUC baseline: **0.762**.
- PR-AUC baseline: **0.247**.
- Brier (calibrated) baseline: **0.068**.

A 3-percentage-point drop is chosen because it exceeds the CV standard deviation by a comfortable margin:

- ROC-AUC σ = 0.004 across the 5 CV folds (from MLflow `std_roc_auc`). A 3 pp drop = 7.5σ — very unlikely to be noise.
- PR-AUC σ = 0.007 across the 5 CV folds. A 3 pp drop = 4.3σ — also clearly signal.

So the threshold is tuned to fire on real degradation, not on the natural run-to-run variance of the metric.

The 0.72 ROC-AUC floor is equivalent to 0.762 − 0.042 ≈ 0.72. It's a slightly tighter threshold than the PR-AUC rule, because ROC-AUC decay tends to be slower and smoother than PR-AUC decay — by the time ROC-AUC falls that far, PR-AUC is usually already 5-10 pp below baseline.

---

## 4. Why sustained, not instantaneous?

A single red-PSI day can be:

- An ETL bug (a column was fed in log-scale one day).
- A downstream system outage masquerading as distributional change.
- A real but transient effect (promotional campaign, holiday traffic).

Sustaining the alert for 7 days filters these out. If the drift is genuine, it persists; if it's noise, it corrects itself within the window.

For concept drift / performance decay, we use a **30-day rolling window** for the same reason: single-day performance can swing widely at low application volumes.

---

## 5. The retrain workflow

When a trigger fires:

### Stage 1 — Triage (hours)

- **Engineer verifies the alert.** Is it a real distributional shift, or an upstream pipeline bug? Check the feature-level PSI breakdown — is it one feature (probably upstream), or many features (probably population)?
- **If it's a bug:** fix the pipeline, reset the monitor, no retrain.
- **If it's real drift:** proceed.

### Stage 2 — Root cause (days)

- **Which families are drifting?** Is it EXT_SOURCE_*, or risk ratios, or demographics?
- **Is the score distribution also drifting?** If score drifts but features don't, suspect concept drift — check the calibration drift monitor (`monitoring/calibration_drift.py`). A widening predicted-vs-observed gap confirms concept drift.
- **Segment performance.** Run `evaluation/segment_analysis.py` on current production scores vs baseline segments. Is the degradation concentrated in one segment (e.g. low-document-count applicants — bin `feat_document_count` into low / medium / high), or global?

### Stage 3 — Retrain (1-2 weeks)

- **Pull a new rolling 6-month window** of data.
- **Re-run leakage scan and schema validation** — never trust the new data silently.
- **Re-run feature selection** (Boruta + VIF). If the selected set has materially changed (Jaccard similarity < 0.6 against the current 15-feature set), flag to model-risk-management.
- **Retrain champion and challenger** with Optuna HPO on the new data.
- **Evaluate** via `scripts/evaluate_and_compare.py`.
- **Recalibrate** thresholds — the new score distribution may not land at the same 0.15 / 0.40 decision gates. Re-run `evaluation/threshold_analysis.py` with the current cost matrix.

### Stage 4 — Promotion gate (days)

- **Run the 4-gate comparison** against the currently-deployed champion (not the original baseline; that's drifted).
- **Fairness** check must pass (AIR ≥ 0.80, no EO/DP regression).
- **Calibration** check against the new production distribution.
- **Canary deploy** to a small % of production traffic. Compare scores and decisions against the old model on the same applications.
- **Full deploy** if canary looks clean for 7 days.

### Stage 5 — Document (always)

- Update `experiments/CHAMPION_CHALLENGER_LOG.md` with the new round.
- Re-issue `governance/model_card.md` with the new metrics, training date, and MLflow run ID.
- Persist the old model for 12+ months — retention required for ECOA audit of adverse actions issued under the old model.

---

## 6. Rolling window vs expanding window

For a credit risk model, **rolling** windows generally beat expanding windows because:

- Borrower behaviour changes over economic cycles; 5-year-old data may be actively misleading.
- Product mix, acquisition channels, and bureau pipelines change over multi-year horizons — stale data encodes stale business reality.

For a cold-start lender or a thin-data segment, an **expanding** window is preferred for the opposite reason: more data beats fresher data when data is scarce.

CreditGuard defaults to a 6-month rolling window (`config/train.yaml::monitoring.retrain.window_months: 6`). Two alternative windows are worth keeping in the dossier:

- **12-month** rolling: smoother, catches seasonal effects, but slower to adapt to regime changes.
- **Expanding** with time-decay weighting: compromise — use all history but weight recent examples higher.

The choice is revisited at each retrain and documented.

---

## 7. Guardrails on automatic retraining

Retrain is **not** fully automatic. The system flags the trigger; a human approves the retrain kickoff, reviews the candidate model via the 4-gate comparison, and approves deployment. Reasons:

1. **Prevent pipeline bugs from cascading.** A bad upstream ETL change can produce a plausible-looking PSI alert. An auto-retrain on bad data poisons the model.
2. **Regulatory defensibility.** SR 11-7 and similar regimes require human review of material model changes. A retrain with new features (or materially different selected feature set) qualifies as a material change.
3. **Calibration volatility.** Retraining with a small rolling window can produce noisy calibration. Human review catches obvious regressions.

---

## 8. Rollback criteria (separate from retrain)

Rollback to the previously deployed model if:

- Rolling PR-AUC drops more than 3 pp from **immediate pre-deploy** baseline (regardless of the original 0.247).
- Rolling Brier increases more than 5 pp.
- Any single-day red-PSI event on a new-model-specific feature (a feature that's in the new model but wasn't in the old one).
- Any decline rate swing > 20% from pre-deploy — even if metrics look fine, a 20% swing in decline rate is operationally disruptive and worth investigating.

Rollback is faster and less consequential than a forward retrain; it buys time for Stage 2 triage.

See `monitoring/champion_challenger.py::rollback_check` for the implementation.

---

## 9. Summary table

| Trigger | Condition | Action |
|---|---|---|
| Sustained red drift | PSI ≥ 0.20 for 7 days | Trigger retrain, human-approved |
| Performance floor | Rolling ROC-AUC < 0.72 OR PR-AUC < 0.217 for 30 days | Trigger retrain, human-approved |
| Calibration drift | Predicted-vs-observed gap > 0.10 on any two deciles for 30 days | Trigger retrain, human-approved |
| Rolling Brier blows out | Above calibrated baseline + 0.05 for 30 days | Investigate first; retrain if confirmed |
| Fairness regression | AIR < 0.80 or EO/DP gap jumps > 0.05 | Investigate + fair-lending review; retrain with mitigations if real |
| Rollback | Post-deploy PR-AUC drops 3pp, Brier rises 5pp, or decline rate swings 20% | Immediate rollback, then triage |

---

## 10. Linked artefacts

- `monitoring/retrain_strategy.py` — the rolling/expanding window implementation.
- `monitoring/alert_manager.py::retrain_trigger` — the trigger evaluator.
- `monitoring/champion_challenger.py::rollback_check` — rollback logic.
- `artifacts/evaluation_results.json` — source of the 0.762 / 0.247 / 0.068 baselines.
- `config/train.yaml::monitoring` — the configured thresholds and window.
