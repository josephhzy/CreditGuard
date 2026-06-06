"""CreditGuard training pipeline.

Submodules:
    train             -- Main config-driven training pipeline
    cv_pipeline       -- Cross-validation strategies (stratified, grouped, temporal)
    feature_selection -- Boruta, RFE, VIF, and hybrid feature selection
    hyperparameter_search -- Optuna-based hyperparameter optimization
    experiments       -- Systematic experiment tracks (1-5) logged to MLflow
"""
