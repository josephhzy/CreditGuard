# artifacts/

This directory contains committed output files required by the Streamlit
dashboard and evaluation scripts. They are derived from the public Kaggle
Home Credit Default Risk dataset and are committed so the dashboard runs
without re-executing the full training pipeline.

---

## File inventory

| File | Contents | Produced by |
|------|----------|-------------|
| `oof_predictions_scored.parquet` | Out-of-fold predictions for all 307,511 applicants — see below | `scripts/evaluate_and_compare.py` |
| `fairness_threshold_sweep.csv` | AIR (adverse-impact ratio) at nine decline thresholds for both sensitive attributes | `scripts/fairness_threshold_sweep.py` |
| `shap_mean_abs.csv` | Mean absolute SHAP values per feature for the champion model | `scripts/evaluate_and_compare.py` |
| `evaluation_results.json` | Champion vs. challenger metrics (AUC, KS, Brier, ECE) | `scripts/evaluate_and_compare.py` |
| `ablation_no_bureau_summary.json` | Ablation metrics when bureau features are withheld | `scripts/evaluate_and_compare.py` |
| `reliability_lightgbm.png` | Calibration curve plot for the champion LightGBM model | `scripts/evaluate_and_compare.py` |
| `lightgbm_model.pkl` | Trained champion LightGBM model (feature names + HPO params baked in) | `training/train.py` |
| `lightgbm_no_bureau.pkl` | Ablation model trained without bureau features | `training/train.py` |
| `xgboost_model.pkl` | Trained challenger XGBoost model | `training/train.py` |

---

## oof_predictions_scored.parquet — column details

| Column | Type | Description |
|--------|------|-------------|
| `SK_ID_CURR` | int | Kaggle applicant ID (public dataset key) |
| `y_true` | int (0/1) | Actual default label from the Kaggle training set |
| `y_score` | float | Champion model out-of-fold predicted probability of default |
| `CODE_GENDER` | str | Applicant gender (M / F; XNA treated as NaN) |
| `NAME_EDUCATION_TYPE` | str | Highest education level reported |

`CODE_GENDER` and `NAME_EDUCATION_TYPE` are included because the Streamlit
dashboard renders live fairness curves (adverse-impact ratio by threshold) and
`scripts/fairness_threshold_sweep.py` reads them directly. Removing these
columns would break both consumers. The values are sourced from the public
Kaggle dataset and do not represent real individuals.

---

## Data provenance

All files are derived from the **Home Credit Default Risk** public dataset:
<https://www.kaggle.com/c/home-credit-default-risk>

No data outside that public release is present in this directory.
