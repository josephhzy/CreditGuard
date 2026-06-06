# Fairness Statement

> **Note:** this file is a short statement. The full measured fairness analysis and its limitations is in `governance/FAIRNESS_REPORT.md`. Measured AIR: gender 0.81 at t=0.40, education 0.61 at t=0.40 (education fails the 80% rule). That report also covers the documented reject-option mitigation.

## Scope

This project includes illustrative fairness diagnostics intended to surface potential disparities in model behaviour across demographic groups. They are not a fair-lending review and do not certify the model for production use.

## What This Is

- Fairness diagnostics applied to the trained model: demographic parity, equalized odds, adverse impact ratio, calibration by group, and proxy sensitivity
- Illustrative scope — not a regulatory fair-lending review or production certification

## What This Is NOT

- A regulatory compliance certification
- An exhaustive fair lending analysis
- A guarantee of non-discriminatory outcomes
- A substitute for qualified legal/compliance review

## Diagnostics Included

1. **Demographic Parity:** Compares approval rates across groups
2. **Equalized Odds:** Compares true positive and false positive rates across groups
3. **Adverse Impact Ratio:** Tests the 80% rule (min group rate / max group rate)
4. **Calibration by Group:** Checks if predicted probabilities are well-calibrated within each group
5. **Proxy Sensitivity:** Measures how much scores change when potentially sensitive features are perturbed

## Limitations of Fairness Checks on Public Data

- Protected attributes in the Home Credit dataset are limited (gender, education level)
- Real fair lending analysis requires access to HMDA-style demographic data
- Intersectional analysis (combinations of protected attributes) is limited by sample sizes
- Proxy variable analysis cannot capture all forms of indirect discrimination
- Fairness metrics can conflict with each other; no single metric captures all notions of fairness

## Production deployment requirements

1. Fair lending counsel review on real protected-class data
2. Disparate impact analysis on actual protected classes
3. Intersectional analysis where sample sizes permit
4. Business justification documented for any observed disparities
5. Ongoing fairness monitoring alongside performance monitoring
6. An appeals process for adverse decisions
