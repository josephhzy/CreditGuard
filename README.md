# CreditGuard

**Credit Default Early Warning and Decisioning System**

> A calibrated LightGBM probability-of-default model with a three-band approve/review/decline layer, batch drift monitoring (PSI / KS / Jensen-Shannon), and a regulated-lender governance stack (model card, fairness report, adverse-action notices, champion-challenger log). Built on the Home Credit Default Risk dataset (Kaggle, 307,511 applications) — a reference implementation of credit decisioning methodology; production deployment would require lender-specific data, calibration, and compliance review.

---

## Results (at a glance)

| | Champion: **LightGBM** | Challenger: **XGBoost** |
|---|---|---|
| ROC-AUC | **0.7621** | 0.7607 |
| PR-AUC (primary) | **0.247** | 0.244 |
| KS statistic | 0.392 | 0.391 |
| Gini | 0.524 | 0.522 |
| Brier (raw) | **0.177** | 0.210 |
| Brier (Platt-calibrated) | 0.068 | 0.068 |
| Lift @ top 1% | 5.77x | 5.79x |
| Features | 15 of ~80 (Boruta + VIF) | 15 of ~80 |

- **Base default rate:** 8.07% (TARGET=1 in Home Credit).
- **Governance decision:** `INVESTIGATE` (expected — governance framework correctly flags the education-AIR failure; champion is retained for reference). Challenger failed PR-AUC (-0.0026) and raw calibration (+0.033 Brier); the champion separately fails the 80% AIR rule on `NAME_EDUCATION_TYPE` (0.6056) and on `CODE_GENDER` at the approve gate (0.71 at t=0.15). Champion is retained by default (challenger did not improve); deployment remains blocked pending the reject-option mitigation documented in `governance/FAIRNESS_REPORT.md` §1. Full comparison in `artifacts/evaluation_results.json` and `experiments/CHAMPION_CHALLENGER_LOG.md`.
- **WoE scorecard baseline** designed as the interpretable baseline; IVs and coefficients on the real dataset not yet computed — see `decisioning/scorecard.py`.
- Evaluation protocol: 5-fold StratifiedGroupKFold (grouped on `SK_ID_CURR`), Optuna HPO (50 trials targeting PR-AUC), honest out-of-fold (OOF) predictions. **Caveat:** because feature engineering aggregates the relational tables to one row per `SK_ID_CURR`, the Group dimension is degenerate at evaluation time — StratifiedGroupKFold is functionally equivalent to StratifiedKFold here. The grouped variant is kept as a guard against any future schema change that re-introduces multi-row applicants. **No out-of-time validation** is performed, because the Home Credit aggregated rows do not carry a clean application-date column; performance on a true future cohort is unknown for this dataset.
- **Top 5 features by measured mean |SHAP|** (log-odds, TreeExplainer on the full 307K training set; full ranking in `artifacts/shap_mean_abs.csv`): (1) `feat_ext_source_mean` 0.4377, (2) `feat_annuity_to_credit` 0.2388, (3) `feat_ext_2x3` 0.1791, (4) `feat_employment_years` 0.1326, (5) `feat_education_ordinal` 0.1245. External-score features (the five `feat_ext_*` variables) concentrate ~46% of total attribution across the 15-feature set — see `features/FEATURE_IMPORTANCE.md` for the full table and interpretation.
- **Bureau-score ablation (held-out, 10 features):** dropping the five `feat_ext_*` features from the champion's 15-feature set reduces ROC-AUC from 0.7621 to 0.7026 (KS 0.3917 → 0.3001). Note: this is a 15 vs 10 feature comparison with no re-selection (the 10-feature variant reuses the champion's hyperparameters without re-running feature selection or HPO) — the delta conflates bureau-score loss with reduced model capacity. See `features/FEATURE_IMPORTANCE.md` section 6 for the asymmetry note.
- **Reliability diagram** (raw LightGBM, 10 uniform bins on OOF predictions) saved at `artifacts/reliability_lightgbm.png`; methodology in `evaluation/CALIBRATION.md`.

> At 8% prevalence the random baseline PR-AUC is ~0.08; the champion's 0.247 is ~3x over base rate with 5.77x lift at the top 1%. Published results above 0.4 PR-AUC on Home Credit have generally involved leakage or data-split violations.

---

## Scope and Out-of-Scope

**What CreditGuard models:** Probability of Default (PD) — the likelihood that an applicant will default on the loan being applied for, at or before the observation horizon used in the Home Credit labels.

**What it deliberately does not model:**

- **Loss Given Default (LGD).** Recovery rates and collateral haircuts.
- **Exposure at Default (EAD).** Outstanding balance at the default event (draw-down behaviour, revolving utilisation).
- **Recoveries / collections workflow.** Default is treated as a terminal state.
- **Pricing, pre-approval, or cross-sell routing.** Decisioning here is binary accept/review/decline.

In a real deployment, PD from a model like this one would feed the standard Basel-style expected-loss decomposition: **Expected Loss = PD × LGD × EAD**. A scorecard that only produces PD is half the credit-risk system; the other half (LGD, EAD, and a policy rules engine) is out of scope for this project.

See `decisioning/COST_MATRIX.md` for how business economics would enter once LGD/EAD are added.

---

## Executive Summary

CreditGuard is a credit-default scoring system built on the Home Credit dataset — 307,511 applications across 7 relational tables, ~8% default rate. It pairs a calibrated LightGBM model with a three-band decisioning layer (approve / review / decline), batch drift monitoring (PSI / KS / Jensen-Shannon), and the categories of governance artefacts a regulated lender would review before promoting a credit model: a model card, a fairness report (with the measured education-AIR failure), an adverse-action document, a cost-matrix derivation, and a champion-challenger log. The pipeline covers schema-validated data ingestion, leakage scanning, ~80 engineered features (count varies with dataset column availability) reduced to 15 via hybrid Boruta + VIF, 5 experiment tracks logged in MLflow, cost-sensitive threshold analysis, and SHAP-based reason codes consistent with ECOA-style explanations. This is a reference implementation; a production deployment would require lender-specific data, re-calibration, and a fair-lending review by qualified counsel.

---

## System Architecture

```
                         CreditGuard System Architecture
 ============================================================================

  Raw Data (7 CSVs)        Feature Matrix          Model Training
 +------------------+    +------------------+    +-------------------+
 | application_train|    | ~80 engineered   |    | 5 experiment      |
 | bureau           |--->| features across  |--->| tracks + Optuna   |
 | bureau_balance   |    | 7 families       |    | HPO + MLflow      |
 | previous_app     |    | (parquet)        |    | logging           |
 | POS_CASH_balance |    +------------------+    +-------------------+
 | installments     |            |                        |
 | credit_card_bal  |    Schema validation        StratifiedGroupKFold (final)
 +------------------+    Leakage scanning         Temporal split (explored, not used — no date column)
         |               Quality reports           Boruta/RFE/VIF
         v                                               |
  Join + Aggregate                                       v
  (star schema to                              +-------------------+
   one-row-per-                                | Evaluation        |
   applicant)                                  | - Discrimination  |
                                               | - Calibration     |
                                               | - Threshold       |
                                               | - Fairness        |
                                               | - Temporal        |
                                               +-------------------+
                                                         |
                    +------------------------------------+
                    |                                    |
                    v                                    v
          +------------------+               +--------------------+
          | Decisioning      |               | Monitoring         |
          | - Approve/Review |               | - PSI drift        |
          |   /Decline bands |               | - KS / JS div.     |
          | - WoE scorecard  |               | - Calibration      |
          | - Business sim   |               |   drift            |
          | - Portfolio      |               | - Alert manager    |
          |   analytics      |               | - Champion vs      |
          +------------------+               |   challenger       |
                    |                        | - Retrain trigger  |
                    v                        +--------------------+
          +------------------+
          | Serving (FastAPI)|
          | /predict         |
          | /predict-batch   |
          | /explain         |
          | /drift/report    |
          | /model-card      |
          | /health          |
          | /metadata        |
          | /threshold/*     |
          +------------------+
```

---

## Quick Start

**Requirements:** Python 3.10 or 3.11 (CI runs against 3.11). On 3.9 or 3.13+ the install will fail or use untested wheels.

```bash
# 1. Clone and install
git clone https://github.com/josephhzy/creditguard.git
cd creditguard
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
pip install -e ".[dev]"

# 2. Download the dataset
# **Prerequisite:** `pip install kaggle` and a configured API token (see comment below), OR be ready to download manually from Kaggle's website.
# Kaggle setup is a one-time step:
#   a. pip install kaggle
#   b. Generate an API token at https://www.kaggle.com/settings ("Create New
#      Token") and place the downloaded kaggle.json at ~/.kaggle/kaggle.json
#      (chmod 600), OR export KAGGLE_USERNAME and KAGGLE_KEY.
#   c. Accept the competition rules at
#      https://www.kaggle.com/c/home-credit-default-risk/rules.
# `make download` falls back to printing manual download instructions if the
# `kaggle` CLI is missing.
make download        # or: python -m data.download

# 3. Validate, join, and build features
make validate        # schema checks + leakage scan
make join            # one-to-many star-schema join -> joined.parquet
make features        # builds ~80 features -> data/processed/features.parquet

# 4. Train the champion model + post-training reconciliation
make train           # config-driven pipeline, logs to MLflow (~33 min)
make evaluate        # OOF eval, Platt calibrator, fairness reconciliation

# 4b. Optional: reproduce all 5 experiment tracks (~30–45 min)
# make experiments   # see the Modelling Strategy section for what each track tests

# 5. Serve the API
make serve           # uvicorn on port 8000 with auto-reload

# 6. Explore the dashboard (no API key, reads artifacts/)
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

**Note on the challenger.** `make evaluate` runs the champion (LightGBM) by default. If `artifacts/xgboost_model.pkl` is not present, the script skips the challenger gracefully and emits a champion-only report. To train the XGBoost challenger as well, run `make train` with a config file that sets `training.model: xgboost`.

> **Reproducibility note.** The trained LightGBM bundle (`artifacts/lightgbm_model.pkl`) is **not committed** to git — `*.pkl` is in `.gitignore` so the trained weights stay out of the source tree. A fresh clone has the small dashboard artifacts (OOF parquet, SHAP CSV, fairness sweep CSV, evaluation JSON, ablation JSON, reliability diagram PNG — everything the dashboard reads) but **does not have a serving-ready model**. The drift report JSON (`monitoring/reports/latest_drift.json`) is not committed; run `make monitor` to generate it. The deterministic path to a model bundle on a fresh clone is steps 2–4 above; expect ~30–35 minutes of wall time on a laptop CPU. If you only want to inspect results, the streamlit dashboard renders fully from the shipped artifacts without the pickle.

The dashboard exposes: governance decision gates, champion vs challenger headline metrics, lift@k, calibration (Brier raw / Platt / isotonic) and reliability diagram, a live threshold simulator on all 307K OOF predictions, AIR sweep for `CODE_GENDER` and `NAME_EDUCATION_TYPE`, SHAP global importance, and the no-bureau ablation delta.

---

## Data Pipeline

The Home Credit dataset is a star schema with 7 tables linked by `SK_ID_CURR` and `SK_ID_PREV`. The data layer handles:

| Step | Module | What It Does |
|------|--------|-------------|
| Download | `data/download.py` | Fetches from Kaggle API with file-size verification; instructions for manual fallback |
| Schema Validation | `data/schema_validator.py` | Checks expected columns, types, and value ranges for all 7 tables before any modelling begins |
| Table Joins | `data/join_tables.py` | Aggregates one-to-many relationships (mean, max, min, sum, count for numerics; mode, nunique for categoricals) with two-level aggregation for grandchild tables (e.g., bureau_balance through bureau to application) |
| Leakage Scanning | `data/leakage_scanner.py` | Blacklists post-outcome columns (SK_DPD, SK_DPD_DEF), flags suspiciously high target correlations (r > 0.95), detects temporal ordering violations, and checks for future-record aggregation |
| Quality Reports | `data/quality_report.py` | Missing-value profiles, duplicate detection, distribution summaries |

**Output:** A single parquet file with one row per `SK_ID_CURR`, ready for feature engineering.

---

## Feature Engineering

Seven feature families produce ~80 features (count varies with dataset column availability), each designed around a distinct credit risk signal. All families use a sentinel value (`-999`) for missing data to preserve the missingness signal for tree-based models.

| Family | Module | Signal | Example Features |
|--------|--------|--------|-----------------|
| Application Profile | `application_profile.py` | Demographics, income, employment, documentation | `feat_log_income`, `feat_income_quintile`, `feat_age_years`, `feat_employment_years`, `feat_education_ordinal` |
| Bureau History | `bureau_history.py` | External credit history from bureaus | `feat_bureau_active_count`, `feat_bureau_active_ratio`, `feat_bureau_max_overdue_amt`, `feat_bureau_CREDIT_TYPE_*` |
| Previous Applications | `previous_apps.py` | Prior loan applications at the institution | `feat_prev_approval_rate`, `feat_prev_refusal_rate`, `feat_prev_total_count`, `feat_prev_amt_spread` |
| Payment Behaviour | `payment_behaviour.py` | Installment payment reliability | `feat_pay_lateness_mean`, `feat_pay_lateness_max`, `feat_pay_missed_count`, `feat_pay_underpayment_ratio` |
| Utilization | `utilization.py` | Revolving credit and POS usage intensity | `feat_cc_utilization_mean`, `feat_cc_peak_utilization`, `feat_pos_dpd_mean`, `feat_cc_atm_drawings_mean` |
| Risk Ratios | `risk_ratios.py` | Cross-table capacity ratios | `feat_debt_burden`, `feat_annuity_to_income`, `feat_goods_to_income`, `feat_bureau_credit_to_income` |
| Temporal Trends | `temporal_trends.py` | Behavioural trajectory (worsening vs improving) | `feat_trend_dpd_delta`, `feat_trend_dpd_worsening`, `feat_trend_util_delta`, `feat_trend_balance_velocity` |

---

## Modelling Strategy

Training follows 5 experiment tracks, each isolating one design decision. All experiments are logged to MLflow for reproducible comparison.

| Track | Question | Variants Tested | Champion Selection |
|-------|----------|-----------------|-------------------|
| 1. Baseline Models | Which algorithm family? | Logistic Regression, Random Forest, XGBoost, LightGBM | Best PR-AUC (imbalanced data — ROC-AUC alone is misleading at 8% prevalence) |
| 2. Imbalance Treatment | How to handle 92/8 class skew? | No treatment, class_weight, SMOTE, ADASYN, EasyEnsemble, BalancedRF, RUSBoost | PR-AUC improvement without sacrificing discrimination |
| 3. Feature Selection | How many features does the model need? | None, Boruta, RFE, VIF, Hybrid (Boruta + VIF) | PR-AUC vs feature count tradeoff; parsimony for interpretability |
| 4. Validation Design | Does the CV design prevent data leakage? | Random stratified, StratifiedGroupKFold (prevents same applicant in train+val), temporal/out-of-time | Lowest gap between CV and OOT performance (overfitting detection) |
| 5. Calibration | Are probabilities trustworthy? | Raw, Platt scaling (sigmoid), Isotonic regression | Lowest Brier score; predicted PD should match observed default rate per decile |

**Final champion:** LightGBM with Optuna HPO (50 trials), hybrid feature selection, StratifiedGroupKFold validation, and Platt calibration. The WoE logistic regression scorecard is retained as the interpretable baseline for governance.

---

## Evaluation Framework

| Dimension | Metrics | Why It Matters |
|-----------|---------|---------------|
| Discrimination | ROC-AUC, PR-AUC, KS statistic, Gini coefficient | Separation of defaulters from non-defaulters. PR-AUC is primary because the 8% base rate makes ROC-AUC overly optimistic. |
| Calibration | Brier score, Platt scaling, isotonic regression, calibration curves | Are predicted probabilities reliable? A model that says 20% PD should see ~20% observed defaults. Critical for threshold-based decisioning and regulatory reporting. |
| Threshold Analysis | Cost-matrix optimization (FN:FP = 10:1 ratio), target recall, precision-recall tradeoff | Placement of the approve/decline thresholds. See `decisioning/COST_MATRIX.md` for the derivation. Config is set to `$600 FP / $6,000 FN` — i.e. EAD × LGD ≈ $10K × 0.60 for FN and lost NIM × term ≈ $600 for FP. The 10:1 ratio is what matters for the optimum; the scale follows Basel-style book-average conventions. |
| Temporal Robustness *(not run — Home Credit aggregated matrix lacks a canonical application-date column; module available in `evaluation/temporal_robustness.py` but not executed)* | Out-of-time validation, fold stability | Does performance hold on future data, or does the model memorize training-period patterns? Performance on a true future cohort is unknown for this dataset. |
| Segment Analysis *(not run — income bracket and employment-band columns require `joined.parquet`, not committed; module available in `evaluation/segment_analysis.py`)* | Performance by gender, education, income bracket, bureau history depth | Performance by subpopulation to identify where the model underperforms. |
| Fairness | Demographic parity, equalized odds, adverse impact ratio (80% rule) | Illustrative bias diagnostics on `CODE_GENDER` and `NAME_EDUCATION_TYPE`. Flags potential disparate impact before deployment review. |

---

## Decisioning Layer

The model produces a raw risk score; the decisioning engine maps that **raw score** to a business action using the bands below. A Platt-calibrated probability of default is also computed and returned to API callers as `risk_score` (see *Score scales* in the API section). The bands were tuned on the raw-score distribution — `decisioning/THRESHOLD_JUSTIFICATION.md` has the derivation.

| Raw score range | Decision | Action |
|---------------|----------|--------|
| 0.00 -- 0.15 | **APPROVE** | Proceed to offer generation |
| 0.15 -- 0.40 | **REVIEW** | Route to manual underwriter for additional review |
| 0.40 -- 1.00 | **DECLINE** | Issue decline notice with adverse action reasons |

Thresholds are configurable in `config/train.yaml` and can be simulated via the `/threshold/simulate` API endpoint before changing production policy. See `decisioning/THRESHOLD_JUSTIFICATION.md` for the rationale behind the 0.15 / 0.40 band and how it moves under different cost assumptions. Cost matrix: **FP=$600, FN=$6,000** (10:1 ratio derived from LGD × EAD economics; see `decisioning/COST_MATRIX.md`). The cost-optimal single-threshold operating point on the champion's OOF predictions is **0.50**. The 0.40 production decline gate is a deliberate deviation from the cost optimum — it encodes (a) real-world cost asymmetries the matrix does not capture (reputational/regulatory cost of FN, downturn LGD), (b) regulatory appetite for FN vs FP under SR 11-7, and (c) robustness across a 4× range of assumed FN:FP ratios. **At the 0.40 gate, 43.8% of applications route to manual review**, meaning the model directly auto-decides 56.2% of the book (14.0% auto-approve + 42.2% auto-decline) — the rest is a human-in-the-loop flag. In a live deployment the 44% review queue would require dedicated underwriting capacity; this prototype treats the queue as costless, which a lender would need to model explicitly. Full review-band sensitivity in `decisioning/THRESHOLD_JUSTIFICATION.md` section 2.

**Decisioning modules:**

- **Decision Engine** (`decisioning/decision_engine.py`) -- score-to-band mapping with SHAP-based adverse action explanations; confidence notes when scores fall near boundaries
- **Business Simulation** (`decisioning/business_simulation.py`) -- project portfolio-level outcomes (approval rates, expected loss, revenue) under different threshold settings
- **Portfolio Analytics** (`decisioning/portfolio_analytics.py`) -- distribution analysis, concentration risk, period-over-period score drift
- **WoE Scorecard** (`decisioning/scorecard.py`) -- Weight-of-Evidence logistic regression producing familiar credit score points (300--850 scale); serves as the interpretable baseline that regulators and underwriters can inspect coefficient-by-coefficient

**Why two models?** Regulated lenders commonly maintain a gradient-boosting model alongside an interpretable scorecard for governance. The scorecard is designed to serve this validation role: comparing its WoE-logistic coefficients against the booster's SHAP rankings would expose divergences that signal data-quality issues or spurious patterns. This comparison has not yet been run on the full dataset; no IV table or coefficient-vs-SHAP summary is committed.

---

## Monitoring and Governance

| Component | Module | Function |
|-----------|--------|----------|
| PSI Drift | `monitoring/psi.py` | Population Stability Index for score and feature distributions. Green < 0.10, Amber < 0.20, Red >= 0.20. |
| Drift Engine | `monitoring/drift_engine.py` | KS test and Jensen-Shannon divergence for feature-level distributional shifts. |
| Calibration Drift | `monitoring/calibration_drift.py` | Tracks whether predicted PDs still match observed default rates over time. |
| Alert Manager | `monitoring/alert_manager.py` | Traffic-light system (GREEN / AMBER / RED) with recommended actions per severity level. |
| Champion-Challenger | `monitoring/champion_challenger.py` | Side-by-side governance framework comparing the production model against a retrained candidate. Promotion requires the challenger to clear configured tolerance bands on PR-AUC, raw Brier, fairness AIR, and interpretability — see `experiments/CHAMPION_CHALLENGER_LOG.md` for the gates and the current `INVESTIGATE` outcome (challenger fails PR-AUC -0.0026 / Brier +0.033; champion separately fails the education-AIR gate). |
| Retrain Strategy | `monitoring/retrain_strategy.py` | Rolling 6-month window retraining. Triggers: performance drop (3% PR-AUC decline), sustained drift alert, or scheduled cadence. |
| Model Card | `governance/model_card.md` | Intended use, non-use policy, assumptions, limitations, ethical considerations, real performance summary (ROC-AUC 0.762, PR-AUC 0.247). |
| Fairness Report | `governance/FAIRNESS_REPORT.md` | Measured AIR on CODE_GENDER (0.81 at decline gate, 0.71 at approve gate) and NAME_EDUCATION_TYPE (0.61 at decline gate). Education AIR fails the 80% rule; gender AIR fails at the approve gate. |
| Adverse Action | `governance/ADVERSE_ACTION.md` | How SHAP-derived reason codes feed ECOA-style notices, and the correlational-not-causal caveat. |
| Retrain Policy | `monitoring/RETRAIN_POLICY.md` | Concrete triggers: PSI > 0.20 sustained 7 days OR rolling ROC-AUC below 0.72. |
| Drift Demo | `monitoring/DRIFT_DEMO.md` | How PSI/KS/JS alert on synthetic shifts; traffic-light mapping. |
| Feature Importance | `features/FEATURE_IMPORTANCE.md` | The 15 selected features, SHAP / ablation methodology. |
| Cost matrix | `decisioning/COST_MATRIX.md` | Derivation of FN=$6,000 and FP=$600 from book economics (EAD × LGD for FN, lost NIM × term for FP). Config updated to these derived values. |
| Calibration | `evaluation/CALIBRATION.md` | Reliability-diagram methodology, Brier scores before/after Platt and isotonic. |
| Champion-challenger log | `experiments/CHAMPION_CHALLENGER_LOG.md` | Gate-by-gate record of LightGBM vs XGBoost (keep-champion, -0.0026 PR-AUC, +0.033 Brier). |

**Adverse action reasons** generated from SHAP attributions are **correlational, not causal**; they reflect what the model used to arrive at the score, consistent with ECOA-style adverse action notices. See `governance/ADVERSE_ACTION.md`.

---

## API Reference

The serving layer is a FastAPI application with structured logging, request ID tracing, and automatic OpenAPI documentation at `/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Score a single applicant. Returns risk score, decision band, top contributing factors, and confidence note. |
| POST | `/predict-batch` | Score up to 1,000 applicants in a single request (batches >500 may cause high latency). |
| POST | `/explain` | Full SHAP explanation for one applicant: base value, per-feature contributions sorted by magnitude, natural-language summary. |
| GET | `/drift/report` | Latest feature drift report with per-feature statistics, overall severity, and recommended actions. |
| GET | `/model-card` | Model governance metadata: intended use, limitations, performance summary, fairness notes. |
| GET | `/health` | Readiness check. Returns 200 with `"status": "ok"` when the model is loaded; 503 with `"status": "degraded"` when no model is loaded so orchestrators (Docker, k8s, load balancers) stop routing traffic to the instance. |
| GET | `/metadata` | Model version, feature count, training date, config hash. |
| GET | `/threshold/recommendation` | Current vs **illustrative** recommended decision thresholds. The recommendation is a heuristic derived from the configured cost matrix; for the production decline gate the cost-optimal value computed on OOF predictions is **0.50**, and the deployed gate (0.40) is the deliberate deviation documented in `decisioning/THRESHOLD_JUSTIFICATION.md`. The endpoint is intended for what-if exploration, not as an authoritative replacement for that analysis. |
| POST | `/threshold/simulate` | Simulate the portfolio impact of proposed threshold changes before applying them. Derives exact band rates and estimated default rates by counting against the 307K OOF predictions in `artifacts/oof_predictions_scored.parquet` (committed). Uses `monitoring/reports/score_distribution.json` instead if that file is present (e.g., after `make monitor`). Returns HTTP 503 if neither file exists. |

Start the server:

```bash
uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Example: API Contract

The model expects the **engineered feature vector** (the 15 selected `feat_*` columns from Boruta + VIF), not raw applicant fields. Feature engineering across bureau history, previous applications, and payment behaviour cannot be reconstructed from a single applicant payload at inference time without a feature store, so the production contract is to send `engineered_features` directly. For demos and reproducibility against the public Home Credit dataset, the API also resolves engineered features by `SK_ID_CURR` lookup against `data/processed/features.parquet`.

**Production request** - caller already has the engineered feature vector from a feature store:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "SK_ID_CURR": 100002,
    "engineered_features": {
      "feat_annuity": 24700.5,
      "feat_credit_to_goods": 1.1583974358974358,
      "feat_employment_years": 1.74,
      "feat_education_ordinal": 1.0,
      "feat_age_bin": 1.0,
      "feat_document_count": 1.0,
      "feat_bureau_BUREAU_CREDIT_TYPE_nunique": 2.0,
      "feat_prev_PREV_NAME_CASH_LOAN_PURPOSE_nunique": 1.0,
      "feat_debt_burden": 2.007888888888889,
      "feat_annuity_to_credit": 0.06074926678103038,
      "feat_ext_source_2": 0.2629485927471776,
      "feat_ext_1x2": 0.021834453721541525,
      "feat_ext_2x3": 0.036648665240279724,
      "feat_ext_source_mean": 0.1617871134127632,
      "feat_ext_source_std": 0.09202580687065609
    }
  }'
```

**Demo request** - only `SK_ID_CURR` is supplied; the API resolves engineered features from the local lookup parquet:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"SK_ID_CURR": 100002}'
```

**Response (real values from the shipped LightGBM bundle, applicant 100002):**

> Note: `risk_score` is the Platt-calibrated PD (population mean ≈ 8%); the 0.15/0.40 decision thresholds operate on the raw model score (mean ≈ 0.38). Do not compare `risk_score` directly against those thresholds — see *Score scales* below.

```json
{
  "sk_id_curr": 100002,
  "risk_score": 0.354408,
  "risk_band": "high",
  "decision": "DECLINE",
  "top_factors": [
    {"feature": "feat_ext_source_mean", "value": 0.1617871134127632, "contribution": 1.189359},
    {"feature": "feat_ext_2x3", "value": 0.036648665240279724, "contribution": 0.515077},
    {"feature": "feat_annuity_to_credit", "value": 0.06074926678103038, "contribution": 0.335596},
    {"feature": "feat_employment_years", "value": 1.74, "contribution": 0.172115},
    {"feature": "feat_ext_1x2", "value": 0.021834453721541525, "contribution": 0.156122}
  ],
  "model_version": "0.1.0",
  "confidence_note": "Prediction is within normal operating parameters."
}
```

**Field semantics:**

1. `risk_score` — the **calibrated probability of default (PD)** from the champion LightGBM: `calibrator(model.predict_proba(x))`, a Platt-scaled true probability (the bundle ships a calibrator fit on out-of-fold predictions). It is a reporting field for the caller (note: this PD lives on a different numeric scale than the 0.15/0.40 decision gates — see *Score scales* below). See `evaluation/CALIBRATION.md` and `scripts/fit_calibrator.py`.
2. `decision` / `risk_band` — computed by `_map_decision()` in `serving/routes/predict.py` from the model's **raw score** against the 0.15 / 0.40 operating points in `config/train.yaml`. Those thresholds were derived on the raw-score distribution (`decisioning/THRESHOLD_JUSTIFICATION.md`, `evaluation/threshold_analysis.py`; cost-optimal raw threshold 0.50). Because Platt calibration is strictly monotonic, the decision is always consistent with the rank-ordering of `risk_score` — a higher PD never receives a more lenient decision.
3. `top_factors` — SHAP values (or LightGBM gain importance, fallback) for the top contributing features, with the raw feature value alongside the SHAP magnitude. Correlational, not causal; see `governance/ADVERSE_ACTION.md`.
4. `confidence_note` — surfaced when the raw score sits near the decline boundary (0.35–0.45; note the window is asymmetric — it fires on both sides of the 0.40 gate, so REVIEW scores near 0.40 also receive it), or when the applicant has a high missing-feature rate, so an underwriter knows the decision is borderline or data-thin. For raw scores at or above 0.45 the note reads "normal operating parameters".

> **Score scales — important.** `risk_score` (a calibrated probability, population mean ≈ the 8% base rate) and the decision thresholds (0.15 / 0.40, on the raw class-balanced score, population mean ≈ 0.38) live on **different numeric scales by design**. Do not compare `risk_score` directly against 0.15 / 0.40 — for applicant 100002 above, `risk_score` is 0.35 yet the decision is `DECLINE`, because the raw score (≥0.40 for this bundle; exact value varies with training) is well past the decline gate. The calibrated-PD equivalents of the gates are ≈0.02 (approve) and ≈0.06 (decline). Because the champion was trained with class-weight balancing, the raw-score distribution is shifted upward relative to a balanced population; the Platt calibrator then compresses predicted PDs toward the 8% base rate, producing a right-skewed calibrated-PD distribution where most applicants cluster well below the mean — so a 6% decline gate still catches the upper tail that accounts for 42% of volume.

**422 contract.** A request with no `engineered_features` and a `SK_ID_CURR` not in the demo lookup is rejected with HTTP 422 and a message that names both alternatives. The API does not silently NaN-fill the feature vector — that previously caused every applicant to score the same number.

---

## Limitations

- **Public dataset.** The Home Credit dataset is a Kaggle competition dataset, not proprietary production data from a regulated lender. Feature distributions and default definitions may not match any specific institution's book.
- **Illustrative fairness.** Fairness diagnostics are demonstrative. They surface potential disparities but do not constitute a regulatory fair lending review. A production deployment requires review by qualified compliance staff.
- **No live feedback loop.** The system monitors drift via distributional statistics but does not receive real-time label feedback. Model decay is detected via proxy metrics, not observed default outcomes.
- **No collections modelling.** Default is treated as a terminal state. A full credit lifecycle would include recovery, cure rates, and loss-given-default estimation.
- **Simplified policy rules.** The three-band decisioning engine demonstrates the pattern but does not encode a full policy rules engine (e.g., DTI hard caps, minimum income floors, product-specific criteria).
- **SHAP explanations are not causal.** Feature contributions indicate correlation direction and magnitude, not causal mechanisms. Adverse action reasons derived from SHAP should be reviewed by underwriters.
- **Single geography/product.** The model is trained on a single lender's population and may not generalize across geographies, products, or time periods without revalidation.

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data | pandas, numpy, pyarrow, pyyaml |
| ML | scikit-learn, LightGBM, XGBoost, imbalanced-learn, Boruta, Optuna |
| Explainability | SHAP |
| API | FastAPI, Uvicorn, Pydantic v2 |
| Experiment Tracking | MLflow |
| Monitoring | Custom (PSI, KS, JS divergence, calibration drift) |
| Visualization | matplotlib, seaborn |
| Logging | structlog |
| Quality | ruff, mypy, pytest, pre-commit |
| Infrastructure | Docker, Make |

---

## Project Structure

```
creditguard/
|-- config/
|   |-- train.yaml              # All pipeline configuration in one place
|   +-- feature_schema.yaml     # Expected feature names for serving
|
|-- data/
|   |-- download.py             # Kaggle dataset acquisition
|   |-- schema_validator.py     # Column/type/range validation
|   |-- join_tables.py          # 7-table star schema join + aggregation
|   |-- leakage_scanner.py      # Post-outcome column detection
|   +-- quality_report.py       # Missing values, distributions, duplicates
|
|-- features/
|   |-- application_profile.py  # Income, age, employment, documentation
|   |-- bureau_history.py       # External credit bureau signals
|   |-- previous_apps.py        # Prior application outcomes
|   |-- payment_behaviour.py    # Installment lateness, missed payments
|   |-- utilization.py          # Card utilization, POS usage
|   |-- risk_ratios.py          # Debt burden, annuity/income, cross-table ratios
|   |-- temporal_trends.py      # Behavioural trajectory / direction of change
|   +-- build_all.py            # Orchestrator: runs all families, quality checks
|
|-- training/
|   |-- train.py                # Config-driven training pipeline
|   |-- cv_pipeline.py          # StratifiedGroupKFold, temporal split
|   |-- feature_selection.py    # Boruta, RFE, VIF, hybrid
|   |-- hyperparameter_search.py # Optuna integration
|   +-- experiments.py          # 5 experiment tracks (baseline, imbalance, FS, CV, calibration)
|
|-- evaluation/
|   |-- discrimination.py       # ROC-AUC, PR-AUC, KS, Gini
|   |-- calibration.py          # Platt, isotonic, Brier, calibration curves
|   |-- threshold_analysis.py   # Cost-matrix optimization
|   |-- temporal_robustness.py  # Out-of-time stability
|   |-- segment_analysis.py     # Subpopulation performance
|   +-- fairness.py             # Demographic parity, equalized odds, AIR
|
|-- decisioning/
|   |-- decision_engine.py      # Score -> Approve/Review/Decline + explanations
|   |-- business_simulation.py  # Portfolio impact projection
|   |-- portfolio_analytics.py  # Distribution, concentration, vintage analysis
|   +-- scorecard.py            # WoE logistic regression (interpretable baseline)
|
|-- monitoring/
|   |-- psi.py                  # Population Stability Index
|   |-- drift_engine.py         # KS test, JS divergence
|   |-- calibration_drift.py    # PD vs observed default rate tracking
|   |-- alert_manager.py        # Green/Amber/Red alerting
|   |-- champion_challenger.py  # Model comparison governance
|   +-- retrain_strategy.py     # Rolling window retrain triggers
|
|-- serving/
|   |-- app.py                  # FastAPI application factory
|   |-- schemas.py              # Pydantic request/response models
|   +-- routes/
|       |-- predict.py          # /predict, /predict-batch
|       |-- explain.py          # /explain (full SHAP)
|       |-- drift.py            # /drift/report
|       |-- model_card.py       # /model-card
|       |-- health.py           # /health, /metadata
|       +-- threshold.py        # /threshold/recommendation, /threshold/simulate
|
|-- governance/
|   |-- model_card.md           # Model card (intended use, limitations, ethics)
|   |-- fairness_statement.md   # Fairness diagnostics — short summary
|   |-- FAIRNESS_REPORT.md      # Measured AIR on CODE_GENDER and NAME_EDUCATION_TYPE
|   +-- ADVERSE_ACTION.md       # SHAP-based reason codes and ECOA caveat
|
|-- experiments/                # Experiment track runners
|-- tests/                      # pytest suite
|-- infra/                      # Dockerfile, docker-compose
|-- mlruns/                     # MLflow tracking store — gitignored; regenerated by make train
|-- Makefile                    # Pipeline commands
+-- pyproject.toml              # Dependencies, tooling config
```
