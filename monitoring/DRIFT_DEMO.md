# Drift Monitoring: Methodology and Synthetic Demonstration

> Scope: this document explains what drift metrics CreditGuard tracks, how the traffic-light mapping is configured, and how to **reproduce** a drift-alert scenario by perturbing a single feature. It does not rely on a real drifted production dataset — that is out of scope for a Kaggle-based reference implementation. The perturbation script below is intentionally simple and designed to turn the traffic light from green to amber to red.

---

## 1. What CreditGuard monitors

All implementation in `monitoring/`. Three complementary drift signals:

| Signal | Module | Catches |
|---|---|---|
| **Population Stability Index (PSI)** | `psi.py::psi_per_feature` | Discrete distributional shift between reference and current populations per feature. |
| **Kolmogorov-Smirnov (KS)** | `drift_engine.py::ks_test` | Continuous-distribution shift on individual features (max \|CDF_ref − CDF_current\|). |
| **Jensen-Shannon divergence (JS)** | `drift_engine.py::js_divergence` | Information-theoretic shift on binned distributions. Bounded in [0, 1], symmetric — useful as a companion to PSI which can blow up when a bin empties. |
| **Score drift** (the score itself, not its inputs) | `psi.py::calculate_psi` on OOF scores | Overall model output-distribution drift, which can happen even if no single feature moves much. |
| **Calibration drift** | `calibration_drift.py` | Divergence between predicted PD and observed default rate over time. This is the **concept drift** monitor — none of PSI/KS/JS catch it. |

Note: PSI (and score/calibration drift) are written to `monitoring/reports/latest_drift.json` by `run_checks.py` and served by `/drift/report`. KS and JS are computed by `drift_engine.detect_drift()` and exercised by `scripts/drift_demo.py` but are not written by the scheduled report (by design — the scheduled report is PSI-only to keep the API schema stable; KS and JS are available as diagnostic tools via `scripts/drift_demo.py` and `drift_engine.detect_drift()`).

---

## 2. Traffic-light thresholds

From `config/train.yaml::monitoring.psi`:

```yaml
psi:
  threshold_green: 0.10
  threshold_amber: 0.20
```

Interpreted by `psi.py::classify_drift`:

| PSI value | Status | Recommended action |
|---|---|---|
| < 0.10 | **GREEN** | No action. |
| 0.10 - 0.20 | **AMBER** | Investigate. Notify model owner; schedule re-evaluation within 7 days. |
| ≥ 0.20 | **RED** | Evaluate retrain; check data pipeline for structural break; freeze or downgrade the score if justified. |

KS and JS do not have formal traffic lights in the same way. `alert_manager.py::generate_alerts` emits independent alerts for each metric: PSI fires at ≥ 0.10 (amber) / ≥ 0.20 (red); KS statistic fires at ≥ 0.10 (amber) / ≥ 0.20 (red); JS divergence fires at ≥ 0.05 (amber) / ≥ 0.10 (red). Overall drift severity in `detect_drift` is determined by PSI classification alone. The three metrics are complementary diagnostics, not a conjunctive gate.

---

## 3. Why use three methods?

| Scenario | PSI behaviour | KS behaviour | JS behaviour |
|---|---|---|---|
| Mean shift, same variance | Moderate PSI (depends on binning) | High KS (clearly shifts CDF) | Moderate JS |
| Same mean, variance inflates | Moderate PSI | Modest KS | High JS |
| Bin becomes empty (rare category disappears) | **PSI blows up** (log of 0 → clipped, still very large) | Minimal effect | Bounded JS |
| Heavy tail appears | Moderate PSI | Strong KS | Strong JS |

No single statistic dominates all scenarios. The alert_manager combines them so that one metric alone doesn't produce either missed drift or false alarms.

---

## 4. Reproducible synthetic drift demo

The committed script at `scripts/drift_demo.py` takes the training feature matrix, perturbs ONE feature, and walks the traffic light from green to red. The perturbation is a mean shift in units of the reference standard deviation.

```bash
python -m scripts.drift_demo --feature feat_debt_burden --shift-sigma 0.5
python -m scripts.drift_demo --feature feat_debt_burden --shift-sigma 1.0
python -m scripts.drift_demo --feature feat_debt_burden --shift-sigma 2.0
```

Source: see `scripts/drift_demo.py`. The implementation is a thin CLI over `monitoring.psi.psi_per_feature` and `monitoring.drift_engine.detect_drift`; it does not touch the model itself.

### Expected behaviour by shift magnitude

Based on the PSI formula `Σ (p_cur - p_ref) × ln(p_cur / p_ref)` over 10 fixed bins of the reference, with `feat_debt_burden` as the target (ref stddev ≈ 2.69):

| Shift (σ units) | Measured PSI on `feat_debt_burden` | Traffic light | Other 14 features |
|---|---|---|---|
| 0.0 (no perturbation) | 0.0001 (chance-level) | GREEN | GREEN |
| 0.5 | 0.049 | GREEN | GREEN |
| **1.0** | **0.175** | **AMBER** | **GREEN** — isolates the shifted feature cleanly |
| 2.0 | 1.287 | RED (hard) | GREEN |

(Numbers sourced from running `python -m scripts.drift_demo --feature feat_debt_burden --shift-sigma {0.0,0.5,1.0,2.0}`; ran against the committed feature parquet with seed 42.) Note that the `ext_source_*` features have sentinel-inflated variance from `-999` missing-value fill and therefore need a larger `--shift-sigma` value before moving the PSI traffic light — the underlying missingness mass absorbs much of the shift.

The non-targeted features stay green, which is exactly the contrast that makes the alert useful: a single-feature structural break (e.g. an upstream ETL change, a change in external bureau pricing, a new product segment) stands out against the rest.

### Running the demo

The script is committed at `scripts/drift_demo.py`. Invoke it as:

```bash
python -m scripts.drift_demo --feature feat_debt_burden --shift-sigma 1.0
python -m scripts.drift_demo --feature feat_annuity --shift-sigma 0.5
```

The output confirms:

1. The shifted feature's PSI crosses the amber or red threshold as `--shift-sigma` grows (see measured numbers above).
2. The other 14 features stay green (their distributions are unchanged).
3. The `detect_drift` report's top-drifters list is dominated by the target feature — its `root_cause_notes` call out the mean-shift direction and magnitude.

---

## 5. Real production drift patterns (what we'd expect to see)

| Pattern | Likely PSI profile | What it means |
|---|---|---|
| Economic regime shift | 3-5 features amber simultaneously (income, employment, debt-burden ratios) | Macro downturn — retrain with downturn data |
| Upstream ETL change | 1 feature red, rest green | Data bug — fix pipeline, do not retrain the model |
| New acquisition channel | Several demographic features amber | New customer mix — retrain with channel flag |
| Bureau pipeline stale | `feat_ext_source_*` red, other features green | Critical — bureau score is the top SHAP feature family. Freeze auto-decline, escalate to bureau vendor. |
| Score PSI red but no feature PSI red | Usually concept drift | Calibration drift monitor (`calibration_drift.py`) should confirm. Retrain. |

The last two rows are the ones operators most care about: feature-level drift monitoring must distinguish "the pipeline is broken" (fix upstream) from "the population has shifted" (retrain).

---

## 6. Wiring drift alerts to action

`alert_manager.py::generate_alerts` emits alerts with feature, metric, value, severity, message, and timestamp. Example output (two alerts for the same feature, one per metric):

```json
[
  {
    "feature": "feat_debt_burden",
    "metric": "psi",
    "value": 0.27,
    "severity": "red",
    "message": "PSI=0.2700 (RED) for feat_debt_burden — exceeds red threshold 0.20",
    "timestamp": "2024-01-15T09:23:41.123456+00:00"
  },
  {
    "feature": "feat_debt_burden",
    "metric": "ks",
    "value": 0.12,
    "severity": "amber",
    "message": "KS=0.1200 (AMBER) for feat_debt_burden — exceeds amber threshold 0.10",
    "timestamp": "2024-01-15T09:23:41.123456+00:00"
  }
]
```

The `retrain_trigger` function in the same module fires a retrain recommendation if any single RED alert is present (default `red_threshold=1`) or if 3 or more AMBER alerts accumulate (`amber_threshold=3`). It evaluates the current alert list only — there is no time-window or sustained-duration logic in the function itself, and no requirement for a co-occurring performance signal. The policy-level guardrails (7-day sustain windows, human-approved retrain kickoff, performance floor checks) are described in `monitoring/RETRAIN_POLICY.md` and are enforced operationally, not in this trigger function.

---

## 7. What's currently missing (honest)

- No real drifted production dataset exists (Kaggle has no live production traffic). The next-best thing would be a temporal split on Home Credit, but the aggregated feature matrix does not carry a clean application-date column.
- Calibration drift has a module but no demo yet.

---

## 8. Linked artefacts

- `monitoring/psi.py` — PSI calculator + traffic-light classifier.
- `monitoring/drift_engine.py` — KS + JS + combined drift report.
- `monitoring/alert_manager.py` — alert generation, retrain trigger logic.
- `monitoring/calibration_drift.py` — concept drift monitor.
- `config/train.yaml::monitoring` — configured thresholds and methods.
- `monitoring/RETRAIN_POLICY.md` — how sustained drift alerts translate to retrain decisions.
