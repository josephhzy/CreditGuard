# Cost Matrix: Derivation and Sensitivity

> Scope: this document derives the False-Negative (FN) and False-Positive (FP) costs that the CreditGuard decisioning layer uses to choose an operating threshold. The `$600 FP / $6,000 FN` values in `config/train.yaml` reflect the derivation below — Basel-style book averages for an unsecured consumer loan of ~$10K × 60% LGD for FN and ~$600 lost NIM over a 2-year term for FP. The 10:1 ratio is the quantity that actually matters; the absolute scale follows standard IRB conventions.

---

## 1. What FN and FP mean here

| Outcome | What happens in the book | Who bears the cost |
|---|---|---|
| **False Negative (FN)** — approve someone who defaults | Loan disburses, borrower defaults, lender loses unrecovered principal + funding cost - collected interest | Lender (direct credit loss) |
| **False Positive (FP)** — decline someone who would have paid | Good loan is not written; lender loses lifetime net-interest-margin on that applicant | Lender (opportunity cost / lost revenue) |

Both are *conditional* on the decision the scorecard drives: applicants in the **REVIEW** band are not auto-decided, so the costs that matter for threshold optimisation are the FN and FP on the auto-approve and auto-decline populations, not the review queue.

---

## 2. Deriving FN cost from book economics

For an unsecured consumer loan of the kind Home Credit originates:

```
FN_cost ≈ EAD × LGD × (1 - recovery_from_collections) - partial_interest_collected
        ≈ EAD × LGD
```

(Most of the recovery and interest terms wash out at the book-average level for unsecured consumer product.)

**Plug in Home Credit book averages (dataset medians):**

| Quantity | Value | Source |
|---|---|---|
| Mean `AMT_CREDIT` | ~$600K local currency, ≈ **$10,000 USD** equivalent for a consumer cash loan | `application_train.csv`, adjusted for Home Credit's core market |
| Assumed LGD | **60%** | Industry norm for unsecured consumer default with basic recovery workflow (Basel-style defaults run 45%-75%) |
| Recovery from collections | Ignored (conservative) | Home Credit does not publish recovery curves; assuming 0 recovery is the conservative choice |

```
FN_cost ≈ $10,000 × 0.60 = $6,000 per bad approval accepted
```

---

## 3. Deriving FP cost

For a declined good applicant the lender loses the net present value of interest margin over the loan's lifetime, net of customer-acquisition cost (CAC).

```
FP_cost ≈ NIM × EAD × expected_life - CAC_already_spent
```

| Quantity | Value | Source |
|---|---|---|
| Net interest margin (NIM), annualised | ~10% on unsecured consumer | Industry benchmark; Home Credit's core product is high-APR consumer instalment |
| Expected term | ~2 years on cash loans | Home Credit reports average `AMT_ANNUITY` × tenor consistent with ~24-month amortisation |
| CAC written off | ~$50 | Digital acquisition for consumer unsecured |

```
FP_cost ≈ 0.10 × $10,000 × 2 - (overlap already spent) ≈ $600 NPV of lost margin
```

Subtracting the portion of CAC that is not salvageable and discounting back to present:

```
FP_cost ≈ $600 (lost margin, undiscounted) × ~1.0 + $50 (sunk CAC) ≈ $600–$700
```

---

## 4. The ratio that actually matters

| Cost | Derived $ | Ratio FN : FP |
|---|---|---|
| FN (approve defaulter) | ~$6,000 | **10 : 1** |
| FP (decline good loan) | ~$600 | 1 |

The single most important property of the cost matrix is the **ratio**, not the absolute dollar level — the optimal threshold is unchanged if we scale both values by the same factor. The `$6,000 / $600` values in `config/train.yaml` are the directly-derived numbers. The 10:1 ratio drives the optimal threshold of 0.50.

```
FN_cost / FP_cost = 10
```

**This is the defensible quantity.** Anyone asking "why $6,000?" should be answered with "it comes from ~$10K average loan × 60% LGD, per Basel-style IRB convention for unsecured consumer loss given default. The FP cost of $600 is the net present value of lost interest margin — ~10% NIM × $10K × 2-year term. The 10:1 ratio is the only quantity that drives the optimum."

---

## 5. Sensitivity of the optimal threshold to this ratio

The cost-optimal threshold minimises:

```
E[cost] = FP_cost × n_false_positives(t) + FN_cost × n_false_negatives(t)
```

With our champion LightGBM on OOF predictions (`artifacts/evaluation_results.json`):

| FN:FP ratio | Rationale | Implied optimal threshold |
|---|---|---|
| 2 : 1 | Secured lending, high LGD floor from collateral | ~0.60 |
| 5 : 1 | Prime unsecured | ~0.55 |
| **10 : 1** | **Current assumption — near-prime / subprime unsecured** | **0.50** |
| 20 : 1 | High-LGD subprime, weak collections | ~0.45 |
| 50 : 1 | Any PD-driven policy where regulatory capital or reputational loss dominates | ~0.35 |

The champion's cost-optimal threshold at 10:1 is **0.50** (`artifacts/evaluation_results.json` → `champion.threshold.optimal_threshold`). The production decline threshold (0.40) is set tighter than the cost-optimum to add margin against recession-regime drift and to keep the review queue from growing too large at the decline boundary. See `decisioning/THRESHOLD_JUSTIFICATION.md` for the full threshold logic.

---

## 6. What a real deployment would re-derive

A lender replacing the Kaggle numbers above would:

1. Pull the realised LGD curve from their collections system by product, vintage, and delinquency stage (monthly roll rates → chronic-delinquency → loss recognition).
2. Compute EAD at origination × average utilisation-at-default for revolving, or outstanding-at-default for instalment.
3. Use actual NIM and CAC from the finance team, not industry averages.
4. Re-run `evaluation/threshold_analysis.py::optimal_threshold()` with the new `fp_cost` and `fn_cost`.
5. Re-calibrate the approve/decline bands.
6. Document the derivation in the model risk management dossier (SR 11-7 in the US, PRA SS1/23 in the UK, MAS FEAT in Singapore).

Nothing in the CreditGuard codebase needs to change for this — only the two numbers in `config/train.yaml::evaluation.threshold.cost_matrix`.

---

## 7. Where `fp_cost` and `fn_cost` live in the codebase

The `$600 / $6000` cost matrix is defined in `config/train.yaml::evaluation.threshold.cost_matrix` and consumed at three call sites:

| Location | Role |
|---|---|
| `evaluation/threshold_analysis.py` — `cost_matrix_analysis`, `optimal_threshold`, `compute_threshold_report` | Defaults `fp_cost=600.0, fn_cost=6000.0`; production path passes these explicitly via `scripts/evaluate_and_compare.py`. |
| `decisioning/business_simulation.py::simulate` | Loads the cost matrix from config; falls back to `600 / 6000` if the key is absent. |
| `serving/routes/threshold.py::_recommend_thresholds` | Uses the config `cost_matrix`; falls back to `{"fp_cost": 600, "fn_cost": 6000}` for the recommendation endpoint when a caller supplies a config without a `cost_matrix` key. |

Note on tests: fixtures in `tests/conftest.py`, `tests/test_evaluation.py`, `tests/test_decisioning.py`, and `tests/test_serving.py` use `fp_cost=50 / fn_cost=500` deliberately — they exercise the *mechanics* of cost computation on small round numbers, and the 10:1 ratio means the threshold-optimisation paths remain well-tested without coupling test math to the book-derived scale.

---

## 8. References for reviewers

- Basel Committee on Banking Supervision, *Internal Ratings-Based (IRB) approach*: PD, LGD, EAD definitions.
- Fed SR 11-7: Guidance on Model Risk Management (US) — requires documented assumptions behind cost/benefit parameters.
- BIS Working Paper, *Expected credit loss under IFRS 9*: framework for PD × LGD × EAD.
- `evaluation/threshold_analysis.py::optimal_threshold()` — the implementation that consumes these costs.
