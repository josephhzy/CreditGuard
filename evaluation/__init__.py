"""CreditGuard evaluation suite.

Submodules:
    discrimination      -- ROC-AUC, PR-AUC, KS statistic, Gini, lift
    calibration         -- Brier score, ECE, Platt scaling, isotonic regression
    threshold_analysis  -- Cost-matrix analysis, approval/bad rate curves, optimal threshold
    temporal_robustness -- Performance by period, OOT validation, score stability
    segment_analysis    -- Subgroup performance across income, employment, education
    fairness            -- Demographic parity, equalized odds, adverse impact ratio, proxy sensitivity
"""
