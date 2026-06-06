# Fairness Report: Honest Scope and Limitations

> Scope: this report states precisely what the CreditGuard fairness diagnostics can and cannot conclude, given that the Home Credit Kaggle dataset does not include protected-class attributes (race, ethnicity) and the 15 Boruta+VIF-selected features do not directly include even the weak proxies (`CODE_GENDER`, `NAME_EDUCATION_TYPE`) that the dataset does contain.

---

## 1. Measured AIR on champion LightGBM

`scripts/evaluate_and_compare.py` joins `CODE_GENDER` (and `NAME_EDUCATION_TYPE`) from `data/processed/joined.parquet` (`XNA` is treated as `NaN`, not as a third group) and passes the result as the optional `sensitive` parameter to `evaluate_model`, which computes AIR, demographic-parity gap, and equalized-odds gap at the **production decline threshold of 0.40** (from `config/train.yaml`). If no sensitive column is available, the code records `NaN` rather than a default value, so an unmeasured result cannot be confused with a passing result downstream.

### Measured AIR on champion LightGBM (OOF predictions, 307,511 rows)

| Slice | Threshold | Groups | AIR (min/max approval) | Passes 80% rule? |
|---|---|---|---|---|
| `CODE_GENDER` (M vs F, XNA excluded) | 0.40 (decline gate) | F n=202,448, M n=105,059 | **0.8125** | yes (narrowly) |
| `CODE_GENDER` | 0.15 (approve gate) | same | **0.7122** | **no (fails)** |
| `CODE_GENDER` | 0.50 (cost-optimum) | same | 0.8591 | yes |
| `NAME_EDUCATION_TYPE` (5 levels) | 0.40 | Lower secondary n=3,816 to Secondary n=218,391 | **0.6056** | **no (fails)** |

The approval-rate split at `t = 0.40` on `CODE_GENDER`:

| Group | n | Approve rate | Default rate | TPR | FPR |
|---|---|---|---|---|---|
| F | 202,448 | 0.618 | 0.070 | 0.745 | 0.355 |
| M | 105,059 | 0.502 | 0.101 | 0.813 | 0.463 |

### What this means

1. **Gender crosses the 80% bar only at the production decline gate and above.** At the tighter approve gate of 0.15, AIR drops to **0.71 on gender**, below the 80% four-fifths floor. At the 0.40 decline gate it is **0.81 on gender** — within the rule but only narrowly, and with a `0.06` TPR gap and a `0.11` FPR gap — both non-trivial (males are more likely to be correctly flagged as defaulters, which is expected given the 0.10 vs 0.07 base-rate gap, but the FPR gap reflects excess false positives on males beyond what the base-rate difference explains).
2. **Education shows a much larger gap** — AIR 0.61 across the five levels. The model approves 74% of Higher-education applicants vs 45% of Lower-secondary, a 29-percentage-point spread. Some of this is driven by real credit-risk differences (Lower-secondary default rate 0.109 vs Higher-education 0.054), but the disparate *impact* remains regardless of the cause — that is what the 80% rule screens for.
3. **Feature-selection did not prevent the disparity.** Even without `CODE_GENDER` or `NAME_EDUCATION_TYPE` in the 15-feature set, the model learns sufficient proxy from `feat_employment_years`, `feat_education_ordinal`, `feat_age_bin`, `feat_document_count`, and the correlated income-signal features to produce meaningfully different approval rates across both slices.
4. **Dataset ceiling applies.** Home Credit has no race/ethnicity columns. The above analysis uses weak proxies; a US deployment would need to re-run with HMDA-matched protected-class data before any fair-lending claim could be made.

### Full threshold sweep

The three-threshold table above covers the operational points (approve gate 0.15, decline gate 0.40, cost-optimum 0.50). To localise where the 80% rule starts to bind, the sweep was extended across the full grid `[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]`. Source: `scripts/fairness_threshold_sweep.py`; raw output at `artifacts/fairness_threshold_sweep.csv`.

| Threshold | AIR CODE_GENDER | Pass 80%? | AIR NAME_EDUCATION_TYPE | Pass 80%? |
|-----------|-----------------|-----------|-------------------------|-----------|
| 0.10      | 0.7177          | no        | 0.1570                  | no        |
| 0.15      | 0.7122          | no        | 0.2115                  | no        |
| 0.20      | 0.7272          | no        | 0.3067                  | no        |
| 0.25      | 0.7418          | no        | 0.3722                  | no        |
| 0.30      | 0.7605          | no        | 0.4611                  | no        |
| 0.35      | 0.7861          | no        | 0.5349                  | no        |
| **0.40**  | **0.8125**      | **yes**   | 0.6056                  | no        |
| 0.45      | 0.8363          | yes       | 0.6618                  | no        |
| 0.50      | 0.8591          | yes       | 0.7201                  | no        |

**What the sweep clarifies:**

1. **Gender crosses the 80% bar only at the production decline gate (t=0.40) and above.** Every tighter threshold — approve gate, review-band interior, any low-cutoff policy — fails the four-fifths rule on gender. The margin at t=0.40 (0.8125) is fragile: a ±0.02 drift in per-group approval rate would move it either side of 0.80.
2. **Education fails the 80% rule at every threshold in the grid, up to and including t=0.50.** The tightest AIR observed (0.16 at t=0.10) shows how severely the score distribution separates Lower-secondary from Higher-education applicants at the risky tail. Even at the cost-optimum of 0.50, AIR is 0.72 — still ~10pts below the four-fifths line.
3. **The approve gate at 0.15 is the most fairness-exposed operating point in the current policy.** Applicants declined at the approve gate (i.e. score ≥ 0.15 → not auto-approved, sent to review) have AIR 0.71 on gender and 0.21 on education. This is where legal exposure is highest; any defence of the current policy would need to show that the review-queue decisioning downstream of the auto-approve step mitigates this.
4. **Monotonicity check.** AIR is approximately monotone increasing in threshold for gender (aside from the 0.10→0.15 inversion from rate compression near zero) and strictly monotone for education — the model's per-group rank ordering is stable, so there is no pathological threshold below which the fairness metric would re-tighten. Mitigation at any single threshold will need per-group or in-processing intervention.

### Mitigation considered

Three options were weighed:

| Option | Mechanism | Cost | Regulatory notes |
|---|---|---|---|
| **Reject-option classification** (post-processing) | For applicants scoring near the decline boundary, flip decisions toward the disadvantaged group's favour within a narrow margin. | **Cheapest** — implemented at decision time, no retrain. Requires a margin parameter and monitoring. | Some US regulators treat post-hoc per-group adjustments as disparate *treatment*; EU GDPR permits with documented justification. Plausible for the REVIEW-band post-processing step, less so at the auto-decline gate. |
| **Threshold-by-group** | Use different t for each sensitive group. | Cheap — two lines of code in `decisioning/decision_engine.py`. | **Generally not allowed under US ECOA** (facially different treatment by protected class). Viable only in jurisdictions that permit such adjustments or as a within-policy reject-inference artefact. |
| **Fairness-constrained retrain** | Optimise a constrained objective (Zafar, Hardt-Price-Srebro, or exponentiated gradient) that penalises disparity during training. | **Most expensive** — full retrain + new HPO, more complex model, reproducibility overhead. | Cleanest from a disparate-treatment standpoint because it intervenes in the learned boundary rather than at decision time. Preferred for fair-lending defence. |

**Recommended direction (not yet implemented): reject-option classification, applied only to the REVIEW band post-processing step**, not at the auto-decline gate. Trade-off: it is the cheapest option and avoids the disparate-treatment exposure of per-group thresholds, but it only moves the margin cases and does nothing for deep-decline or deep-approve scores; it also requires a fair-lending counsel sign-off on the margin parameter. Documented as the governance choice pending the real-data fairness rerun described in section 4.

---

## 2. What is computed — methodology

### 2.1 Demographic parity

`evaluation/fairness.py::demographic_parity`

Compares the approval rate (proportion of applicants with predicted PD below the approve gate) across groups of the sensitive attribute. A gap of 0 means identical approval rates; a gap above ~0.05 typically warrants investigation.

```
DP_gap = max_g approval_rate_g - min_g approval_rate_g
```

### 2.2 Equalized odds

`evaluation/fairness.py::equalized_odds`

Compares true-positive rate (TPR, catching defaulters) and false-positive rate (FPR, declining non-defaulters) across groups. Independently computes gaps on both:

```
EO_TPR_gap = max_g TPR_g - min_g TPR_g
EO_FPR_gap = max_g FPR_g - min_g FPR_g
```

Equalized odds is a harder bar than demographic parity — it requires the model to be equally *accurate* across groups, not just equally *permissive*.

### 2.3 Adverse impact ratio (80% rule)

`evaluation/fairness.py::adverse_impact_ratio`

```
AIR = min_g approval_rate_g / max_g approval_rate_g
```

Under US EEOC guidance (four-fifths rule), AIR < 0.80 is the bright line that presumptively suggests adverse impact. This rule was adapted into fair-lending practice (e.g. ECOA, Reg B, HMDA analysis) but is not a legal safe harbor — it is a *flag*.

### 2.4 Calibration by group

`evaluation/fairness.py::calibration_by_group`

Computes Brier and reliability curve separately for each group. A model that is globally calibrated but miscalibrated within a subgroup will send systematically more (or fewer) of that group into the review band.

### 2.5 Proxy sensitivity

`evaluation/fairness.py::proxy_sensitivity`

Shuffles a sensitive column across all rows and re-scores. If the score distribution moves, the model is responding to the sensitive column (directly or via highly correlated proxies). If it doesn't move, the sensitive column is either genuinely absent from the effective feature set or its signal is fully captured by non-sensitive features.

---

## 3. Which protected-attribute proxies exist in Home Credit

The Home Credit dataset does not contain legally protected-class attributes under US or EU regulations directly. It contains a small set of weak proxies:

| Column | What it is | Protected-class correlation | How it would be handled in production |
|---|---|---|---|
| `CODE_GENDER` | M/F/XNA | Direct proxy for sex | Regulated attribute in EU under GDPR + Gender Directive; under ECOA in US, sex is a protected class. Must not be used as a direct feature in most US lending contexts. |
| `NAME_EDUCATION_TYPE` | Education level | Correlates with age, income, socioeconomic status, race (via school access) | Generally allowable but must be monitored for disparate impact. |
| `NAME_FAMILY_STATUS` | Marital status | Protected class under ECOA | Must not be used directly. |
| `CNT_CHILDREN` / `CNT_FAM_MEMBERS` | Family composition | Can proxy for family-status discrimination | Sensitive. |
| `NAME_HOUSING_TYPE` / `REGION_*` | Address / region | Can proxy for race via redlining history | Heavily scrutinised in US fair lending. |
| `DAYS_BIRTH` → age | Age | Age is protected class (ECOA, ADEA) | Monitored; generally allowable if statistically justified. |
| `OCCUPATION_TYPE` / `ORGANIZATION_TYPE` | Job | Can proxy for sex and ethnicity | Standard underwriting input, but must be monitored. |

**None of these are in the final 15-feature set** — they were either not engineered or dropped by Boruta/VIF. Any production deployment on a real lender's data should re-run fairness with:

1. The **full** engineered feature set, not just the 15 selected — to check that the selector is not concentrating disparate impact into "safe-sounding" proxies.
2. **Real protected-class labels** (HMDA-linked, or consented demographic capture) rather than weak proxies.
3. **Intersectional slices** (e.g. gender × age × education).
4. **A reject-inference step** so the fairness analysis isn't conditioned on the historical approval process.

---

## 4. What the pipeline would do with real protected-class data

Assuming a production deployment has access to actual protected-class data (joined externally, e.g. HMDA-matched), the workflow would be:

1. Update `config/train.yaml::evaluation.fairness.sensitive_features` to include the real attribute (race, ethnicity).
2. Re-run `evaluation/fairness.py` on OOF predictions — it already handles arbitrary categorical sensitive attributes.
3. Compute the four metrics above per group.
4. If AIR < 0.80 or EO gap > 0.05, feed into the **mitigation decision tree**:
   - Stage 1: inspect SHAP by group — which features are driving the disparity?
   - Stage 2: pre-processing (reweighing, disparate impact remover) if disparity is in the input distribution.
   - Stage 3: in-processing (constrained optimisation, e.g. Zafar/Hardt post-hoc calibration) if disparity is in the learned decision boundary.
   - Stage 4: post-processing (threshold-adjusted decisioning per group) if regulatory regime allows it. In the US, threshold-adjusted lending by protected attribute is generally **not** allowed — the tuning must be done upstream via features or training.
5. Document outcomes in the model validation dossier alongside a fair-lending counsel sign-off.

The infrastructure is in place in `evaluation/fairness.py`; the missing pieces are data access and legal review.

---

## 5. Explicit limitations of the current fairness result

1. **Feature selection does not include the sensitive attributes directly.** AIR measured on this model reflects the proxy signal carried by the selected features.
2. **Kaggle data is not representative.** Home Credit is a single lender's historical book, subject to the biases of that lender's prior approval policy (survivorship bias: only approved historical applicants have observed default outcomes — reject inference is not performed here). Because applicants historically declined by Home Credit — disproportionately those with lower income, lower education, or higher predicted risk — have no observed outcome, the default rates and approval rates for those sub-groups are measured on a selected, lower-risk sub-population; the measured AIR gap is therefore likely an underestimate of the true disparity on the full applicant population.
3. **Proxy sensitivity was not run on the shipped model.** The `proxy_sensitivity` function exists in `evaluation/fairness.py` but the evaluate script does not call it on every run; deferred because the weak proxies (`CODE_GENDER`, `NAME_EDUCATION_TYPE`) are absent from the final 15-feature set (either not engineered or dropped by Boruta/VIF), making proxy sensitivity most relevant as a stress-test of whether the selector encoded them via interaction terms — a required step before any production deployment. A proper governance pass would re-run it against the full ~80-feature set to stress-test the selector.
4. **No intersectional analysis.** Demographic parity is computed marginally, not jointly.
5. **No reject inference.** The dataset contains only applications Home Credit chose to fund; declines are absent. A model trained on approvals-only will systematically underestimate default risk for the distributions the historical policy declined.

---

## 6. Status

| Dimension | Status |
|---|---|
| Fairness diagnostics **code** | Implemented, tested, and documented |
| Fairness metrics on **this dataset** | Measured (CODE_GENDER AIR 0.81 at t=0.40, 0.71 at t=0.15; NAME_EDUCATION_TYPE AIR 0.61 at t=0.40). See section 1. |
| Production **fairness certification** | Not claimed, not attempted |
| **Mitigation pipeline** | Methodology documented. The "Mitigation considered" subsection of section 1 commits to Stage 4 — reject-option classification at the REVIEW band post-processing step — as the chosen mitigation direction. The measured gender AIR of 0.71 at the approve gate and education AIR of 0.61 at the decline gate require the upstream diagnostic work in stages 1-3 of the pipeline in section 4 (SHAP-by-group inspection, reweighing, constrained-optimisation experiments) before that chosen mitigation can be finalised and signed off. |

---

## 7. Linked artefacts

- `evaluation/fairness.py` — implementation of all four metrics plus proxy sensitivity.
- `scripts/evaluate_and_compare.py::evaluate_model` — the call site where AIR is computed via `compute_fairness_report` with a joined `CODE_GENDER` column.
- `artifacts/oof_predictions_scored.parquet` — OOF score + sensitive-attribute join used to reproduce the numbers in section 1.
- `artifacts/fairness_threshold_sweep.csv` — AIR at each of nine thresholds for both sensitive attributes. Produced by `scripts/fairness_threshold_sweep.py`.
- `scripts/fairness_threshold_sweep.py` — the sweep script.
- `config/train.yaml::evaluation.fairness` — the configured `sensitive_features` and the `tolerance: 0.80` threshold.
- `governance/model_card.md` — where this report is cited in the ethical-considerations section.
- `governance/ADVERSE_ACTION.md` — adverse-action reason code generation is related but separate from fairness measurement.
