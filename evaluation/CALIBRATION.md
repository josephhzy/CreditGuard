# Calibration: Methodology and Observed Results

> Scope: this document explains the calibration methodology CreditGuard uses, reports the measured raw-vs-calibrated results on the champion LightGBM and challenger XGBoost, and documents the reliability diagram for the champion model. Real numbers are from `artifacts/evaluation_results.json`.

---

## 1. Role of calibration in CreditGuard

Discrimination metrics (ROC-AUC, PR-AUC, KS, Gini) answer: *can the model rank-order defaulters above non-defaulters?*

Calibration metrics (Brier, ECE, reliability diagram) answer: *when the model says 20% PD, do ~20% of those applicants actually default?*

Calibration enters the pipeline in two places:

1. **The reported `risk_score` field is a calibrated PD.** The API returns the Platt-scaled probability so a caller reading `risk_score = 0.12` can treat it as "12% probability of default" — directly meaningful for reporting, capital calculations, or stress-testing.
2. **The champion-challenger calibration gate uses raw Brier.** Calibration is what separated the two candidates here: both models have near-identical ROC-AUC (0.762 vs 0.761), but raw Brier diverges (0.177 vs 0.210), and that gap was the binding constraint in the governance comparison.

**The decision bands themselves are on the raw score, not the calibrated PD.** The 0.15 / 0.40 cutoffs in `config/train.yaml` apply to the raw class-balanced model output (see `decisioning/THRESHOLD_JUSTIFICATION.md`); calibration surfaces a true probability in the response. The two roles are decoupled by design: bands stay where the threshold analysis derived them, and the calibrated PD is returned as a separate, interpretable field.

---

## 2. Metrics computed

All from `evaluation/calibration.py`.

| Metric | Definition | What a good value looks like |
|---|---|---|
| **Brier score** | `mean((predicted_prob - actual_outcome)^2)` — MSE of probabilities | Lower is better. For 8% prevalence, a model that always predicts 0.08 scores ~0.074. A model that always predicts 0.5 scores 0.25. |
| **ECE (Expected Calibration Error)** | Weighted average of per-bin \|predicted_rate − observed_rate\|, over 10 equal-width bins | Lower is better. ECE of 0 = perfectly calibrated. |
| **Reliability diagram** | Scatter of (mean predicted PD in bin, observed default rate in bin) for each of 10 deciles | Points should lie on the y = x diagonal. |
| **Platt-calibrated Brier** | Brier after fitting a logistic regression (`LogisticRegression(C=1e6)`) from raw score → outcome, and applying it | Should drop below raw Brier. |
| **Isotonic-calibrated Brier** | Brier after fitting `IsotonicRegression` from raw score → outcome | Should be ≤ Platt, at the cost of needing more data to avoid overfit. |

---

## 3. Observed results (source: `artifacts/evaluation_results.json`)

*Note: raw ECE reflects score inflation from `scale_pos_weight=10.41`, not miscalibration of the decision boundary; the pipeline serves the Platt-scaled score (Brier 0.068).*

| Metric | Champion (LightGBM) | Challenger (XGBoost) | Delta (XGB − LGBM) |
|---|---|---|---|
| Brier (raw) | **0.177** | 0.210 | **+0.033** |
| ECE (10 bins, raw) | 0.300 | 0.346 | +0.046 |
| Brier (Platt) | 0.0678 | 0.0679 | +0.0001 |
| Brier (isotonic) | 0.0676 | 0.0677 | +0.0001 |

**Findings:**

1. **Raw calibration gap is the governance signal.** XGBoost's raw Brier is higher by +0.033 and the ECE by +0.046 (both from the table above). This is the exact gap the champion-challenger framework's calibration gate flags (tolerance: Brier degradation < 0.02; observed: +0.033 > 0.02 → gate **fails**).
2. **Post-calibration convergence.** After Platt or isotonic scaling, both models land within 0.0001 of each other on Brier. So given a calibration layer, XGBoost would be *usable*, but the pipeline serves the underlying model; a raw-calibrated LightGBM is strictly safer than a Platt-patched XGBoost in a threshold-based decisioning regime because threshold rules are set on the underlying distribution.
3. **Raw Brier of 0.177 for LightGBM is not great in absolute terms.** A constant 8.07% prediction would score ~0.074. The raw score distribution from an `scale_pos_weight=10.41` LightGBM is pushed away from the base rate toward the positive class — it is implicitly trained as an imbalanced classifier, not a calibrated PD estimator. This is **why** Platt scaling is part of the production pipeline: the raw score rank-orders well but its values are not directly PD.
4. **Both models converge post-calibration because the underlying rank-ordering is nearly identical** (ROC-AUC delta 0.001). Calibration is recovering PD-scale from a good ranker; the ranker quality determines post-calibration Brier.

---

## 4. Reliability diagram (persisted)

The reliability diagram is saved at `artifacts/reliability_lightgbm.png`. It overlays raw LightGBM and Platt-scaled LightGBM on the same axes, both computed on the full 307,511 OOF predictions with 10 uniform bins.

**Measured curve (raw LightGBM, mean-predicted → observed-default):**

| Predicted-PD bin (mean) | Observed default rate |
|---|---|
| 0.068 | 0.009 |
| 0.153 | 0.020 |
| 0.250 | 0.032 |
| 0.348 | 0.051 |
| 0.448 | 0.078 |
| 0.548 | 0.112 |
| 0.648 | 0.158 |
| 0.746 | 0.232 |
| 0.838 | 0.361 |
| 0.915 | 0.606 |

The raw curve shows a uniform over-prediction pattern: `scale_pos_weight=10.41` pushes all scores toward the positive class, so the raw curve lies entirely above the diagonal rather than crossing it. The model systematically **over-predicts** PD across every bin. At predicted 0.92 the observed rate is 0.61. This is exactly what `scale_pos_weight=10.41` does — it inflates positive-class probabilities to favour recall over calibration. Under Platt scaling the curve compresses onto the diagonal (the measured Platt points fall within ~0.03 of the y=x line across the filled bins).

Script to reproduce (15 lines):

```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
df = pd.read_parquet("artifacts/oof_predictions_scored.parquet")
y_true, y_score = df["y_true"].values.astype(int), df["y_score"].values
lr = LogisticRegression(C=1e6).fit(y_score.reshape(-1, 1), y_true)
y_platt = lr.predict_proba(y_score.reshape(-1, 1))[:, 1]
pt_r, pp_r = calibration_curve(y_true, y_score, n_bins=10, strategy="uniform")
pt_p, pp_p = calibration_curve(y_true, y_platt, n_bins=10, strategy="uniform")
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", label="Perfect")
ax.plot(pp_r, pt_r, "o-", label="LightGBM raw")
ax.plot(pp_p, pt_p, "s-", label="LightGBM Platt-scaled")
ax.set(xlabel="Mean predicted probability (bin)", ylabel="Observed default rate (bin)")
ax.legend(); plt.savefig("artifacts/reliability_lightgbm.png", dpi=120)
```

OOF predictions are persisted at `artifacts/oof_predictions_scored.parquet` (produced by `scripts/evaluate_and_compare.py` + fairness-join step) so the diagram can be regenerated without re-running the 5-fold CV.

---

## 5. Which calibration method should be deployed?

| Method | Pros | Cons | Chosen? |
|---|---|---|---|
| **Platt (sigmoid)** | 2 parameters, fits on a small validation set, robust | Assumes sigmoid miscalibration shape; underfits if reality is non-monotone | **Yes, primary.** Brier 0.068 on 307K OOF samples. |
| **Isotonic** | Non-parametric, exactly fits piecewise-constant calibration | Needs more data (otherwise steps overfit), can be non-monotone near tails | Alternate. Brier 0.068, so no measured gain over Platt here. |
| **Beta** | 3 parameters, can capture asymmetric miscalibration | Not implemented in current pipeline | No. |

The Platt result saturates here — isotonic is within 0.0001 — so Platt is the deployed method. In a deployment where more labelled data is available a re-evaluation would be warranted.

---

## 6. Calibration drift monitoring (production concern)

`monitoring/calibration_drift.py` tracks predicted PD vs observed default rate across production cohorts. The concern is that a model perfectly calibrated at train time can de-calibrate in production even if the feature distributions don't move — this is **concept drift** (the PD-outcome relationship itself changes). PSI will not catch it. See `monitoring/DRIFT_DEMO.md` and `monitoring/RETRAIN_POLICY.md`.

---

## 7. Linked artefacts

- `artifacts/evaluation_results.json` — the Brier/ECE numbers cited in section 3.
- `artifacts/reliability_lightgbm.png` — the persisted reliability diagram for the LightGBM champion (raw + Platt).
- `artifacts/oof_predictions_scored.parquet` — the OOF predictions + sensitive-attribute join, sufficient to regenerate the diagram.
- `evaluation/calibration.py` — `compute_calibration_report`, `brier_score`, `expected_calibration_error`, `calibrate_platt`, `calibrate_isotonic`.
- `monitoring/calibration_drift.py` — production-time calibration monitoring.
- `governance/model_card.md` — where these numbers flow upward into the model card.
