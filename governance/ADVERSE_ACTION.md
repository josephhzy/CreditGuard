# Adverse Action Notices: SHAP Attributions and the Correlational Caveat

> Scope: this document describes how CreditGuard turns per-applicant SHAP values into adverse-action-style reason codes, why those reasons are **correlational not causal**, and what a production compliance pipeline would add on top. It is the written version of the ECOA-style caveat that also appears inline in the `README.md` and in `governance/model_card.md`.

---

## 1. The correlational caveat

> **Adverse action reasons from CreditGuard are generated from SHAP attributions. These are correlational, not causal; they reflect what the model used to arrive at the score, consistent with ECOA-style explanations.**

This caveat applies to all reason-code output in this project, the README, the model card, and the `/explain` API response.

---

## 2. The regulatory context

Under the US Equal Credit Opportunity Act (ECOA) and its implementing Regulation B, a creditor that takes adverse action against an applicant must provide **specific reasons** for the adverse action (or notice of the right to request them). The Federal Reserve's official Regulation B commentary (12 CFR 1002) requires the reasons to be "the principal reasons for the adverse action" and to be "specific" rather than generic.

Analogous regimes exist in other jurisdictions:

- **UK / PRA:** CONC rules and guidance on fair treatment of rejected applicants require meaningful explanations, though less prescriptive than ECOA.
- **EU / GDPR Art. 22:** When automated decisions produce legal effects, data subjects have a right to "meaningful information about the logic involved."
- **Singapore / MAS FEAT:** Fairness, ethics, accountability, transparency principles with explanation as a practical requirement under the Transparency pillar.

SHAP, applied to gradient-boosted trees, is the industry-standard way to produce the feature-level specificity these regimes require. But SHAP is **attribution**, not **causation**.

---

## 3. What SHAP computes

For each applicant and each feature, SHAP computes a value ϕᵢ such that:

```
predicted_score = base_score + Σᵢ ϕᵢ
```

ϕᵢ is the *contribution of feature i* to this applicant's deviation from the base score, averaged over all possible feature orderings (Shapley values from cooperative game theory). The sign tells you direction (increased or decreased PD); the magnitude tells you weight.

**What SHAP does not do:**

- **It does not establish causation.** A feature with a positive ϕᵢ is one the *model used* to increase the model's raw risk score. It is not necessarily the *reason* the applicant is likely to default.
- **It does not identify counterfactuals.** "If this feature were different, the outcome would have been different" is a separate question requiring intervention-based reasoning (e.g. DiCE, counterfactual explanations). SHAP is observational.
- **It is not unique.** Different reasonable decompositions of the score are possible (TreeSHAP vs PermutationSHAP vs LIME vs IntegratedGradients). We use TreeSHAP because it is exact for tree ensembles and fast enough for online inference.
- **It inherits the model's biases.** If the model has learned a spurious correlation, SHAP will faithfully attribute the score to the spurious feature.

---

## 4. How CreditGuard generates reason codes

Two code paths produce SHAP-derived reason codes, both backed by the same `shap.TreeExplainer` cached at startup:

1. **`/predict`** (`serving/routes/predict.py::_compute_top_factors`) — returns the top-K features by **absolute** SHAP value as a flat list. Each entry keeps the signed `contribution`, so a downstream consumer can split into increases / decreases by the sign of `contribution` if needed.
2. **`/explain`** (`serving/routes/explain.py::_compute_shap`) — returns the **full** per-feature SHAP vector plus the model's base value, suitable for reconstructing the score additively.

The helper in `decisioning/decision_engine.py::explain_decision` (a Python function, not an HTTP route) accepts pre-computed SHAP values and wraps them into a separate dictionary structure with explicit `increases_risk` / `decreases_risk` lists. It exists for offline reason-code work; it is not what the API returns.

Partial `/predict` response for applicant 100002 (decision and factors only; full response shape in README.md):

```json
{
  "decision": "DECLINE",
  "risk_score": 0.72,
  "top_factors": [
    {"feature": "feat_ext_source_mean",    "value": 0.162, "contribution":  1.189},
    {"feature": "feat_ext_2x3",            "value": 0.037, "contribution":  0.515},
    {"feature": "feat_annuity_to_credit",  "value": 0.061, "contribution":  0.336},
    {"feature": "feat_employment_years",   "value": 1.740, "contribution":  0.172},
    {"feature": "feat_ext_1x2",            "value": 0.022, "contribution":  0.156}
  ]
}
```

`risk_score` is the calibrated probability of default (a reporting field); the decision is determined by the raw model score, which is ≥ 0.40 for this applicant — see the Score scales callout in README.md for why these two numbers are on different numeric scales. The `value` field in each `top_factors` entry is the applicant's raw engineered feature value (e.g. 0.162 is the feat_ext_source_mean score from the bureau, not the model's risk score).

A negative `contribution` would indicate a feature pulling the score down (decreases risk). The split into increases / decreases is a one-line filter on the consumer side rather than a structural feature of the response — that keeps the API neutral and lets reason-code mapping live in the compliance layer.

---

## 5. From SHAP features to human-readable reason codes

The `/explain` response produces feature names like `feat_debt_burden`. Those are not sendable in a letter. A production pipeline wraps them in a reason-code dictionary:

| Internal feature | ECOA-style reason code | Human wording (illustrative) |
|---|---|---|
| `feat_debt_burden` | HIGH_LEVERAGE | "Your requested loan amount is high relative to your stated income." |
| `feat_annuity_to_credit` | PAYMENT_PACE | "The periodic repayment burden is high relative to the principal of this loan." |
| `feat_credit_to_goods` | OVER_FINANCING | "The requested loan exceeds the price of the goods being financed." |
| `feat_ext_source_2`, `feat_ext_source_mean`, `feat_ext_source_std` | EXTERNAL_SCORE | "Your external credit bureau score is below our approval threshold." |
| `feat_employment_years` | EMPLOYMENT_TENURE | "Your current employment tenure is shorter than our typical approval profile." |
| `feat_age_bin` | AGE | (ECOA-sensitive — must map carefully; age is a protected class but can be used if actuarially justified) |
| `feat_document_count` | THIN_FILE | "Limited documentation was provided with your application." |
| `feat_bureau_BUREAU_CREDIT_TYPE_nunique` | CREDIT_HISTORY_BREADTH | "Limited diversity of external credit products in your file." |

The mapping is maintained by compliance, not engineering. It is explicitly out of scope for the current implementation — the point here is that the SHAP output is the **raw material** that such a mapping would consume, not the final letter copy.

---

## 6. Rules the compliance pipeline adds

A production adverse-action workflow would wrap `/explain` in at least these rules:

1. **Protected-class scrubbing.** Never emit a reason code that directly references a protected attribute (gender, marital status, national origin). Even if `CODE_GENDER` appears in SHAP, it must not be surfaced as a reason.
2. **Minimum count.** Regulation B requires at least one specific reason on adverse action notices; best practice is to surface the top 4-5 to give the applicant a fuller picture of what was weighed.
3. **Actionability.** The reason should be something an applicant could, in principle, change (or at least plan around). "Your score from an external bureau was too low" is acceptable; "the statistical interaction between your bureau scores was unfavourable" is not.
4. **Audit trail.** Every scored decision must persist the score, the SHAP top-K, the model version, and the config hash — so the specific reason given to an applicant today can be reproduced years later in a regulatory review, even if the model has since been retrained.
5. **Human review for declines.** The underwriter must confirm the decline and the reason codes before the letter goes out. SHAP provides the candidate reasons; humans approve them.

CreditGuard implements step 4 via the structured `/predict` response, the `model_version` field, and a structured `audit_decision` log event (`serving/routes/predict.py`) that captures score, decision, feature source, config hash, and top reasons (SHAP top-5) per request. Durable persistence (writing to a database or append-only store) is not implemented; in production the serving stack can route the `audit_decision` log to a separate sink by filtering on `event="audit_decision"`. Steps 1, 2, 3, and 5 are production-stage compliance tasks, not ML-engineering tasks.

---

## 7. Causal alternatives that a research extension could add

If a future iteration wanted to move beyond correlational SHAP toward causal reasons:

- **Counterfactual explanations** (DiCE, Alibi): "Here is the minimal change to your feature values that would flip the decision." This is closer to causal, still not fully so.
- **Causal forests / double ML:** Estimate the causal effect of specific interventions (e.g. paying down debt) on default risk, using a structural assumption.
- **Structural credit models:** PD → EL pipelines that embed a credit-event model rather than a pure classifier.

None are currently required for ECOA-style compliance. All are desirable for **actionable** advice to rejected applicants.

---

## 8. Linked artefacts

- `decisioning/decision_engine.py::explain_decision` — per-applicant SHAP top-K.
- `serving/routes/explain.py` — `/explain` endpoint.
- `governance/model_card.md` — where this caveat is surfaced for governance reviewers.
- `governance/FAIRNESS_REPORT.md` — adjacent concerns on protected-class attribution.
- `features/FEATURE_IMPORTANCE.md` — global SHAP ranking that complements per-applicant SHAP.
