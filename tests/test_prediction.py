"""
Tests for src/prediction.py
"""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.model_trainer import ModelTrainer
from src.prediction import PredictionEngine
from src.preprocessing import Preprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def iris_engine():
    """Return a fitted PredictionEngine for Iris (classification)."""
    df = DataLoader.load_sample_dataset("Iris (Classification)")
    p = Preprocessor()
    X, y_raw = p.prepare_features(df, "target")
    y, le = p.encode_target(y_raw, "classification")
    pipe = p.build_pipeline(X)

    trainer = ModelTrainer(pipe, "classification", selected_models=["Random Forest"], cv_folds=3, label_encoder=le)
    result = trainer.train_all(X, y)["Random Forest"]

    engine = PredictionEngine(
        pipeline=result.pipeline,
        feature_names=result.feature_names_in,
        task_type="classification",
        label_encoder=le,
        training_data=df,
    )
    return engine, result.feature_names_in, df


@pytest.fixture(scope="module")
def diabetes_engine():
    """Return a fitted PredictionEngine for Diabetes (regression)."""
    df = DataLoader.load_sample_dataset("Diabetes (Regression)")
    p = Preprocessor()
    X, y_raw = p.prepare_features(df, "target")
    y, le = p.encode_target(y_raw, "regression")
    pipe = p.build_pipeline(X)

    trainer = ModelTrainer(
        pipe, "regression",
        selected_models=["Random Forest Regressor"],
        cv_folds=3,
    )
    result = trainer.train_all(X, y)["Random Forest Regressor"]

    engine = PredictionEngine(
        pipeline=result.pipeline,
        feature_names=result.feature_names_in,
        task_type="regression",
        label_encoder=None,
        training_data=df,
    )
    return engine, result.feature_names_in, df


# ---------------------------------------------------------------------------
# Classification prediction
# ---------------------------------------------------------------------------
class TestClassificationPrediction:
    def test_predict_returns_label(self, iris_engine):
        engine, feature_names, df = iris_engine
        sample = {feat: float(df[feat].median()) for feat in feature_names if feat in df.columns}
        output = engine.predict(sample)
        assert output.errors == []
        assert output.predicted_label is not None
        assert isinstance(output.predicted_label, str)

    def test_predict_returns_probabilities(self, iris_engine):
        engine, feature_names, df = iris_engine
        sample = {feat: float(df[feat].median()) for feat in feature_names if feat in df.columns}
        output = engine.predict(sample)
        assert output.probabilities is not None
        total = sum(output.probabilities.values())
        assert abs(total - 1.0) < 1e-4

    def test_all_classes_in_probabilities(self, iris_engine):
        engine, feature_names, df = iris_engine
        sample = {feat: float(df[feat].median()) for feat in feature_names if feat in df.columns}
        output = engine.predict(sample)
        assert len(output.probabilities) == 3  # Iris has 3 classes

    def test_missing_features_trigger_warning(self, iris_engine):
        engine, feature_names, df = iris_engine
        incomplete = {}  # provide no features
        output = engine.predict(incomplete)
        assert len(output.warnings) > 0
        assert any("missing" in w.lower() or "imputation" in w.lower() for w in output.warnings)

    def test_out_of_range_value_triggers_warning(self, iris_engine):
        engine, feature_names, df = iris_engine
        sample = {feat: float(df[feat].median()) for feat in feature_names if feat in df.columns}
        # Set one value to an extreme outlier
        first_feat = feature_names[0]
        sample[first_feat] = 999999.0
        output = engine.predict(sample)
        assert any("percentile" in w.lower() or "outside" in w.lower() for w in output.warnings)


# ---------------------------------------------------------------------------
# Regression prediction
# ---------------------------------------------------------------------------
class TestRegressionPrediction:
    def test_predict_returns_numeric(self, diabetes_engine):
        engine, feature_names, df = diabetes_engine
        sample = {feat: float(df[feat].median()) for feat in feature_names if feat in df.columns}
        output = engine.predict(sample)
        assert output.errors == []
        assert output.predicted_value is not None
        assert isinstance(float(output.predicted_value), float)

    def test_no_probabilities_for_regression(self, diabetes_engine):
        engine, feature_names, df = diabetes_engine
        sample = {feat: float(df[feat].median()) for feat in feature_names if feat in df.columns}
        output = engine.predict(sample)
        # Regression engines should not produce class probabilities
        assert output.probabilities is None

    def test_predicted_label_is_none_for_regression(self, diabetes_engine):
        engine, feature_names, df = diabetes_engine
        sample = {feat: float(df[feat].median()) for feat in feature_names if feat in df.columns}
        output = engine.predict(sample)
        assert output.predicted_label is None


# ---------------------------------------------------------------------------
# build_input_row
# ---------------------------------------------------------------------------
class TestBuildInputRow:
    def test_produces_single_row_dataframe(self, iris_engine):
        engine, feature_names, df = iris_engine
        values = {feat: 1.0 for feat in feature_names}
        row = engine.build_input_row(values)
        assert isinstance(row, pd.DataFrame)
        assert len(row) == 1

    def test_missing_features_are_nan(self, iris_engine):
        engine, feature_names, df = iris_engine
        row = engine.build_input_row({})  # empty dict
        assert row.isna().all(axis=None)

    def test_row_has_all_feature_columns(self, iris_engine):
        engine, feature_names, df = iris_engine
        row = engine.build_input_row({})
        for feat in feature_names:
            assert feat in row.columns
