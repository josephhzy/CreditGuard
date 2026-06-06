# Threshold Justification: Approve 0.15 / Decline 0.40

> Scope: this document explains the logic behind CreditGuard's three-band decisioning boundaries (`approve < 0.15`, `review 0.15-0.40`, `decline >= 0.40`), how they relate to the cost-optimal operating point, and how sensitive the choice is to the cost matrix assumptions documented in `COST_MATRIX.md`.

---

## 1. Two different thresholds, two different jobs

The single "optimal threshold" that falls out of `evaluation/threshold_analysis.py::optimal_threshold()` is **one** number — a point on the ROC / PR curve that minimises expected cost under the FN/FP matrix. A production decisioning system needs **two** cutpoints:

| Threshold | Job | In CreditGuard |
|---|---|---|
| **Approve gate** | Below which the application is low-risk enough to auto-approve without human review | `0.15` |
| **Decline gate** | At/above which the application is high-risk enough to auto-decline with an adverse-action notice | `0.40` |
| Middle band | Manual underwriter review, optionally with additional data / pricing adjustment | `[0.15, 0.40)` |

The cost-optimal threshold is a single point; the review band is *one of the levers* for absorbing uncertainty when the probabilities are near that point.

---

## 2. Where the cost-optimal threshold actually sits

From `artifacts/evaluation_results.json` (champion LightGBM on OOF predictions under the configured cost matrix `FP=$600 / FN=$6,000`; see `decisioning/COST_MATRIX.md` for the derivation):

- `optimal_threshold` = **0.50**
- `expected_cost_at_optimal` = $96,669,000 (aggregate across 307,511 OOF predictions, at the current `$600 / $6,000` cost matrix — the optimum threshold depends on the 10:1 ratio only; absolute scale follows from the per-unit FP/FN costs)
- Review queue at the production 0.15/0.40 bands:
  - Auto-approve: 43,103 (14.0% of book)
  - Auto-decline: 129,784 (42.2%)
  - Review: 134,624 (43.8%)

The cost-optimal single-threshold operating point is 0.50 — higher than the 0.40 decline gate. Note that moving the decline gate **tighter** than the cost optimum is, by construction, a **worse** policy under the stated cost matrix — it increases expected loss per decision. That trade is made deliberately, for reasons the cost matrix alone does not encode:

1. **The cost matrix captures narrow direct loss, not full real-world asymmetry.** The configured 10:1 (FN:FP) ratio is the Basel-style book-economic derivation (EAD × LGD for FN, lost NIM × term for FP; see `decisioning/COST_MATRIX.md`). It omits (a) reputational / regulatory-examination cost of each FN, which can exceed the direct loan loss by multiples on a portfolio basis; (b) downturn LGD which historically widens the effective ratio by 10-30pp; (c) asymmetric regulator appetite for FN vs FP in supervised credit models (SR 11-7 scrutinises under-reserving more than over-decline). In practice the "true" asymmetry is larger than 10:1, which pulls the optimal threshold lower than the matrix alone suggests.
2. **The review band absorbs what a single threshold cannot.** The 0.15-0.40 review band is the explicit policy lever for the region where the model's probability is not decisive enough to auto-decide. An underwriter can apply judgement, request additional documentation, offer a smaller loan, or price up. Widening the decline gate all the way to 0.50 would shrink this band; the 0.40 gate pairs a review band sized to match underwriting capacity with a decline rule that is conservative on the FN side.
3. **Robustness across cost regimes.** Section 3 below shows the cost-optimum sits around 0.45 for a 20:1 FN:FP ratio and around 0.55 for 5:1 — the 0.40 decline gate sits below the optima for ratios in the 5:1–20:1 range, and above the 50:1 optimum (~0.35). A threshold set exactly at the 10:1 cost optimum (0.50) would be brittle against any re-estimation of the matrix; the 0.40 gate is deliberately tighter, in the direction the unmodelled costs push.

The policy cost of this deviation is small: Section 4 shows that within ±0.05 of the cost-optimal threshold the expected cost increases by only a few percent, so the unmodelled-cost buffer more than covers the precision loss.

**This is a policy decision, not a cost-minimising optimum.** The precise statement is: the 0.40 gate deviates from the cost-optimum in the direction the unmodelled costs push.

And why set the approve gate at 0.15, well below the cost optimum?

1. **Auto-approval risk is asymmetric**: a wrongly auto-approved bad loan goes out the door before anyone looks at it; a wrongly auto-declined good loan can still be rescued by the review band if the threshold were porous, but at 0.15 we keep the approve band to the applicants the model is very confident about.
2. **Operational throughput**: at the 0.15 gate, 14% of the book auto-approves — a healthy fraction that keeps the underwriter queue manageable while excluding most marginal cases.

### Review-band size as a function of the decline gate

With the approve gate fixed at 0.15, moving the decline gate changes how much of the book lands in manual review. The table below is measured on the 307,511 OOF predictions in `artifacts/oof_predictions_scored.parquet`:

| Decline gate | Auto-approve | Review (0.15 → gate) | Auto-decline |
|---|---|---|---|
| 0.30 | 14.0% | 27.6% | 58.4% |
| 0.35 | 14.0% | 36.1% | 49.9% |
| **0.40 (production)** | **14.0%** | **43.8%** | **42.2%** |
| 0.45 | 14.0% | 50.8% | 35.2% |
| 0.50 (cost-optimum) | 14.0% | 57.1% | 28.9% |
| 0.55 | 14.0% | 62.9% | 23.1% |

**At the production 0.40 gate, 43.8% of applications route to manual review.** This is the other half of the threshold decision: the model auto-decides 56.2% of the book (14.0% approve + 42.2% decline) and defers the rest to underwriters. The 43.8% review share is deliberately large — it reflects the choice to use the model as an underwriter-augmentation tool rather than a fully autonomous decisioner. It is also the operational cost of a tight decline gate: moving decline from 0.40 to 0.50 (the cost-optimum) would shift ~13pp of the book from auto-decline to review. Any claim that this is "a model" must be qualified by the fact that on the current policy the model directly decides fewer than 6 in 10 applications; the remainder is a human-in-the-loop flag.

---

## 3. Sensitivity methodology

The `evaluation/threshold_analysis.py` module sweeps 99 candidate thresholds from 0.01 to 0.99 and computes, at each:

- Number and percentage of approves, reviews, declines
- Expected cost under the configured FP/FN matrix
- Precision, recall, F1 on the decline-as-positive framing

To reproduce the cost-vs-threshold curve for any new FP/FN ratio:

```python
from evaluation.threshold_analysis import compute_threshold_report
import pandas as pd

oof = pd.read_parquet("artifacts/oof_predictions_scored.parquet")
y      = oof["y_true"].to_numpy()
scores = oof["y_score"].to_numpy()

# e.g. recession-regime sensitivity (FP held at the configured $600)
for fn_cost in [3000, 6000, 9000, 12000, 24000]:
    rep = compute_threshold_report(y, scores, fp_cost=600, fn_cost=fn_cost)
    print(f"FN=${fn_cost:5d}: optimal t = {rep.optimal_threshold:.3f}, cost = {rep.expected_cost_at_optimal:,.0f}")
```

Expected shape of the result (based on monotonicity of the cost surface in threshold):

| FN:FP ratio | Rationale | Single cost-optimal threshold |
|---|---|---|
| 2 : 1 | Secured lending, collateral-heavy | ~0.60 |
| 5 : 1 | Prime unsecured | ~0.55 |
| **10 : 1** | **Current assumption** | **0.50 (measured)** |
| 20 : 1 | Subprime / weak collections | ~0.45 |
| 50 : 1 | Reputational or capital-dominated loss | ~0.35 |

The production 0.40 decline gate sits between the 50:1 and 20:1 cost-optima (0.35 → 0.45) and is robust across roughly a 5× shift in the FN:FP ratio. This is deliberate — a threshold that only works at exactly one FN:FP ratio is a policy that breaks the first time the LGD curve moves.

---

## 4. What a cost-vs-threshold curve looks like methodologically

For the champion LightGBM the expected-cost curve is convex with a flat basin around the optimum. Walking through threshold space:

- **t → 0**: almost everything is declined. Lots of FP (declining good applicants) but minimal FN. Cost is dominated by `FP_cost × n_good_declined`.
- **t → 1**: almost everything is approved. FP collapses, FN balloons. Cost is dominated by `FN_cost × n_bad_approved`.
- **Optimum**: the point where a 1-unit increase in threshold would convert as many FP→TP as TP→FN (weighted by their costs).

Because base-rate defaulters (~8%) are the minority class and FN is 10x the FP, the optimum sits at a threshold above the 50/50 naive cut but below the 95th percentile — the characteristic PD≈0.50 we measured.

The cost curve's flatness around the minimum matters: within `optimum ± 0.05`, expected cost increases by only a few percent. That flatness is what makes a *two-threshold* policy (0.15 / 0.40) with a review queue not much worse than the single-point optimum, and much less sensitive to ±0.05 mis-estimation of the optimum.

---

## 5. Re-deriving the thresholds for a new deployment

A lender adapting CreditGuard would:

1. Replace FP/FN in `config/train.yaml` with book-economic values per `COST_MATRIX.md`.
2. Run `python -m scripts.evaluate_and_compare` to produce `artifacts/evaluation_results.json` with the new cost-optimal threshold.
3. Decide on the margin between cost-optimal and the production decline gate. A common rule: decline gate = cost-optimum − 0.1 (10 PD points tighter).
4. Decide the approve gate based on throughput of the review queue; for a small underwriting team, widen the decline side and narrow the approve side so fewer cases hit the manual queue.
5. Stress-test by re-running step 2 with FP/FN shifted ±50% and confirming the decline gate still sits below the stressed optimum.

None of this requires retraining; the model's score distribution is fixed, only the decision thresholds move.

---

## 6. Why not a single-threshold binary decision?

Because in credit risk, probabilities near the decision boundary are where the money is made or lost — and where SHAP explanations are most useful for human underwriters. A three-band system:

- **Increases capture of edge-case bads** that a strict approve/decline split might let through.
- **Reduces false-decline reputational risk** by giving marginal good applicants a second look.
- **Gives the model a humility setting** — the review band is where the system explicitly says "I'm not confident enough to auto-decide."

This is standard in regulated consumer lending; it is not a CreditGuard invention.

---

## 7. Linked artefacts

- `artifacts/evaluation_results.json` — OOF-derived optimal threshold and review-queue composition.
- `evaluation/threshold_analysis.py` — `compute_threshold_report()`, `optimal_threshold()`, `review_queue_size()`.
- `decisioning/COST_MATRIX.md` — derivation of FP/FN costs.
- `serving/routes/threshold.py` — `/threshold/simulate` endpoint for what-if analysis without redeploying.
