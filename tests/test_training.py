"""Tests for the training layer: CV, feature selection, pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from training.cv_pipeline import get_cv_splits
from training.feature_selection import boruta_select, rfe_select, vif_filter

# ═══════════════════════════════════════════════════════════════════════════
# Cross-validation
# ═══════════════════════════════════════════════════════════════════════════


class TestCVPipeline:
    """Tests for cross-validation split strategies."""

    def test_cv_stratified_preserves_class_ratio(
        self,
        sample_features_df: pd.DataFrame,
    ) -> None:
        """Stratified splits should approximately preserve the positive rate in each fold."""
        splits, report = get_cv_splits(
            sample_features_df,
            strategy="stratified",
            n_splits=3,
            target_col="TARGET",
            group_col="SK_ID_CURR",
        )
        overall_rate = sample_features_df["TARGET"].mean()
        for fold_info in report.folds:
            assert abs(fold_info.train_positive_rate - overall_rate) < 0.05
            assert abs(fold_info.val_positive_rate - overall_rate) < 0.10

    def test_cv_grouped_no_leakage(
        self,
        sample_features_df: pd.DataFrame,
    ) -> None:
        """Grouped CV must not place the same SK_ID_CURR in both train and val."""
        splits, _ = get_cv_splits(
            sample_features_df,
            strategy="stratified_group",
            n_splits=3,
            target_col="TARGET",
            group_col="SK_ID_CURR",
        )
        ids = sample_features_df["SK_ID_CURR"].values
        for train_idx, val_idx in splits:
            train_ids = set(ids[train_idx])
            val_ids = set(ids[val_idx])
            assert train_ids.isdisjoint(val_ids), "Applicant ID leaked across folds"


# ═══════════════════════════════════════════════════════════════════════════
# Feature selection
# ═══════════════════════════════════════════════════════════════════════════


class TestFeatureSelection:
    """Tests for feature selection methods."""

    @staticmethod
    def _make_classification_df(
        n: int = 300,
        p: int = 10,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Create a small classification dataset for selection tests."""
        rng = np.random.default_rng(42)
        X = pd.DataFrame(
            rng.standard_normal((n, p)),
            columns=[f"f{i}" for i in range(p)],
        )
        y = pd.Series(rng.binomial(1, 0.15, size=n), name="TARGET")
        return X, y

    def test_feature_selection_boruta_does_not_crash(self) -> None:
        """Boruta should run end-to-end on random data without crashing.

        The name reflects what the test actually verifies. Boruta on
        signal-free random data may legitimately select zero features, so
        asserting non-emptiness here would be flaky. A separate test with
        injected signal would be needed to verify selection quality.
        """
        X, y = self._make_classification_df()
        selected, report = boruta_select(X, y, max_iter=20, random_state=42)
        assert isinstance(selected, list)
        assert report.method == "boruta"
        assert report.n_input_features == X.shape[1]

    def test_feature_selection_boruta_detects_signal(self) -> None:
        """Boruta should select at least one feature when strong signal exists."""
        rng = np.random.default_rng(0)
        n = 400
        signal0 = rng.standard_normal(n)
        signal1 = rng.standard_normal(n)
        signal2 = rng.standard_normal(n)
        noise = rng.standard_normal((n, 7))
        X = pd.DataFrame(
            np.column_stack([signal0, signal1, signal2, noise]),
            columns=["s0", "s1", "s2"] + [f"noise{i}" for i in range(7)],
        )
        # Deterministic target: majority vote of thresholded signal features
        y = pd.Series(
            ((signal0 > 0).astype(int) + (signal1 > 0).astype(int) + (signal2 > 0).astype(int) >= 2).astype(int),
            name="TARGET",
        )
        selected, report = boruta_select(X, y, max_iter=50, random_state=42)
        signal_features = {"s0", "s1", "s2"}
        assert len(set(selected) & signal_features) >= 1, (
            f"Boruta should select at least one signal feature; got {selected}"
        )

    def test_feature_selection_rfe_returns_features(self) -> None:
        """RFE should select exactly the requested number of features."""
        X, y = self._make_classification_df()
        n_target = 5
        selected, report = rfe_select(X, y, n_features=n_target, step=2, random_state=42)
        assert len(selected) == n_target
        assert report.method == "rfe"

    def test_feature_selection_vif_removes_correlated(self) -> None:
        """VIF filter should drop a perfectly collinear feature."""
        rng = np.random.default_rng(42)
        n = 200
        base = rng.standard_normal(n)
        X = pd.DataFrame(
            {
                "a": base,
                "b": base * 2.0 + 0.001 * rng.standard_normal(n),  # collinear with a
                "c": rng.standard_normal(n),
            }
        )
        selected, report = vif_filter(X, threshold=10.0)
        assert report.method == "vif"
        # At least one of (a, b) should be dropped
        assert len(selected) < X.shape[1]


# ═══════════════════════════════════════════════════════════════════════════
# Training pipeline (mock-heavy integration)
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainPipeline:
    """Lightweight smoke test for the training pipeline."""

    @patch("training.train.mlflow")
    @patch("training.train.pd.read_parquet")
    @patch("training.train._build_model")
    def test_train_pipeline_runs_without_error(
        self,
        mock_build_model: MagicMock,
        mock_read_parquet: MagicMock,
        mock_mlflow: MagicMock,
        sample_features_df: pd.DataFrame,
        tmp_path,
    ) -> None:
        """Smoke test: pipeline completes end-to-end with mocked MLflow + small data.

        Heavy mocking (MLflow, parquet read, model builder) means this only
        verifies the code path runs without raising — not that it learned
        anything. The fixture has 200 rows with random features at ~8%
        positive class, so on n_splits=2 the AUC is noisy and any non-trivial
        bound would be flaky. A pipeline that learns something useful is
        verified end-to-end via the `make evaluate` chain on real data.
        """
        import yaml
        from sklearn.linear_model import LogisticRegression as LR

        from training.train import train_pipeline

        # Mock parquet read to return our fixture data
        mock_read_parquet.return_value = sample_features_df.copy()

        # Mock _build_model to return a real LogisticRegression
        mock_build_model.side_effect = lambda model_type, params: LR(
            random_state=42,
            max_iter=1000,
        )

        # Use a dummy path for features (won't actually be read from disk)
        features_path = tmp_path / "features.parquet"
        features_path.touch()

        config = {
            "seed": 42,
            "data": {
                "features_path": str(features_path),
                "target_column": "TARGET",
                "id_column": "SK_ID_CURR",
            },
            "feature_selection": {"method": "none"},
            "training": {
                "model": "logistic",
                "cv": {"strategy": "stratified", "n_splits": 2},
                "hyperparameter_search": {"enabled": False},
            },
            "mlflow": {
                "tracking_uri": str(tmp_path / "mlruns"),
                "experiment_name": "test",
            },
        }
        config_path = tmp_path / "train.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Mock MLflow calls
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        result = train_pipeline(config_path)
        assert result.model_type == "logistic"
        assert result.mean_roc_auc > 0.0
        assert result.n_folds == 2
