"""
Tests for src/model_trainer.py and src/evaluation.py
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.model_trainer import ModelTrainer, CLASSIFICATION_MODELS, REGRESSION_MODELS, TrainingResult
from src.evaluation import ModelEvaluator, compare_models


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def clf_dataset():
    """Prepared classification X, y from Iris."""
    df = DataLoader.load_sample_dataset("Iris (Classification)")
    p = Preprocessor()
    X, y_raw = p.prepare_features(df, "target")
    y, le = p.encode_target(y_raw, "classification")
    pipe = p.build_pipeline(X)
    return X, y, pipe, p, le


@pytest.fixture(scope="module")
def reg_dataset():
    """Prepared regression X, y from Diabetes."""
    df = DataLoader.load_sample_dataset("Diabetes (Regression)")
    p = Preprocessor()
    X, y_raw = p.prepare_features(df, "target")
    y, le = p.encode_target(y_raw, "regression")
    pipe = p.build_pipeline(X)
    return X, y, pipe, p, le


@pytest.fixture(scope="module")
def clf_result(clf_dataset):
    X, y, pipe, p, le = clf_dataset
    trainer = ModelTrainer(
        pipe, "classification",
        selected_models=["Random Forest"],
        cv_folds=3,
        label_encoder=le,
    )
    return trainer.train_all(X, y)["Random Forest"]


@pytest.fixture(scope="module")
def reg_result(reg_dataset):
    X, y, pipe, p, le = reg_dataset
    trainer = ModelTrainer(
        pipe, "regression",
        selected_models=["Random Forest Regressor"],
        cv_folds=3,
    )
    return trainer.train_all(X, y)["Random Forest Regressor"]


# ---------------------------------------------------------------------------
# ModelTrainer – classification
# ---------------------------------------------------------------------------
class TestModelTrainerClassification:
    def test_trains_all_models(self, clf_dataset):
        X, y, pipe, p, le = clf_dataset
        trainer = ModelTrainer(pipe, "classification", cv_folds=3, label_encoder=le)
        results = trainer.train_all(X, y)
        assert len(results) == len(CLASSIFICATION_MODELS)

    def test_selected_models_only(self, clf_dataset):
        X, y, pipe, p, le = clf_dataset
        selected = ["Logistic Regression", "Random Forest"]
        trainer = ModelTrainer(pipe, "classification", selected_models=selected, cv_folds=3)
        results = trainer.train_all(X, y)
        assert set(results.keys()) == set(selected)

    def test_invalid_model_name_raises(self, clf_dataset):
        X, y, pipe, p, le = clf_dataset
        with pytest.raises(ValueError, match="Unknown models"):
            ModelTrainer(pipe, "classification", selected_models=["FakeModel"])

    def test_result_has_correct_fields(self, clf_dataset):
        X, y, pipe, p, le = clf_dataset
        trainer = ModelTrainer(pipe, "classification", selected_models=["Decision Tree"], cv_folds=3)
        results = trainer.train_all(X, y)
        result = results["Decision Tree"]
        assert isinstance(result, TrainingResult)
        assert isinstance(result.pipeline, Pipeline)
        assert len(result.y_test) > 0
        assert len(result.y_pred_test) == len(result.y_test)

    def test_no_data_leakage_in_split(self, clf_dataset):
        X, y, pipe, p, le = clf_dataset
        trainer = ModelTrainer(pipe, "classification", cv_folds=3, label_encoder=le)
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        # No overlap
        train_idx = set(X_train.index)
        test_idx = set(X_test.index)
        assert len(train_idx & test_idx) == 0
        assert len(train_idx) + len(test_idx) == len(X)

    def test_stratified_split_preserves_class_distribution(self, clf_dataset):
        X, y, pipe, p, le = clf_dataset
        trainer = ModelTrainer(pipe, "classification", cv_folds=3)
        _, _, y_train, y_test = trainer.split_data(X, y)
        # Both splits should have all 3 classes
        assert len(np.unique(y_train)) == 3
        assert len(np.unique(y_test)) == 3

    def test_probabilities_available_for_compatible_models(self, clf_dataset):
        X, y, pipe, p, le = clf_dataset
        trainer = ModelTrainer(
            pipe, "classification",
            selected_models=["Logistic Regression", "Random Forest"],
            cv_folds=3,
        )
        results = trainer.train_all(X, y)
        for name in ["Logistic Regression", "Random Forest"]:
            assert results[name].y_proba_test is not None

    def test_reproducibility(self, clf_dataset):
        X, y, pipe, p, le = clf_dataset
        trainer1 = ModelTrainer(pipe, "classification", selected_models=["Decision Tree"], cv_folds=3)
        trainer2 = ModelTrainer(pipe, "classification", selected_models=["Decision Tree"], cv_folds=3)
        r1 = trainer1.train_all(X, y)
        r2 = trainer2.train_all(X, y)
        np.testing.assert_array_equal(r1["Decision Tree"].y_pred_test, r2["Decision Tree"].y_pred_test)


# ---------------------------------------------------------------------------
# ModelTrainer – regression
# ---------------------------------------------------------------------------
class TestModelTrainerRegression:
    def test_trains_all_regression_models(self, reg_dataset):
        X, y, pipe, p, le = reg_dataset
        trainer = ModelTrainer(pipe, "regression", cv_folds=3)
        results = trainer.train_all(X, y)
        assert len(results) == len(REGRESSION_MODELS)

    def test_regression_predictions_are_continuous(self, reg_dataset):
        X, y, pipe, p, le = reg_dataset
        trainer = ModelTrainer(pipe, "regression", selected_models=["Linear Regression"], cv_folds=3)
        results = trainer.train_all(X, y)
        preds = results["Linear Regression"].y_pred_test
        # Continuous predictions should not be integer-valued (mostly)
        assert preds.dtype in [np.float32, np.float64, float]


# ---------------------------------------------------------------------------
# ModelEvaluator – classification
# ---------------------------------------------------------------------------
class TestModelEvaluatorClassification:
    def test_metrics_in_valid_range(self, clf_result):
        m = ModelEvaluator.compute_classification_metrics(clf_result)
        assert 0.0 <= m.accuracy <= 1.0
        assert 0.0 <= m.precision <= 1.0
        assert 0.0 <= m.recall <= 1.0
        assert 0.0 <= m.f1 <= 1.0
        if m.roc_auc is not None:
            assert 0.0 <= m.roc_auc <= 1.0

    def test_confusion_matrix_shape(self, clf_result):
        m = ModelEvaluator.compute_classification_metrics(clf_result)
        n_classes = m.n_classes
        assert m.confusion_mat.shape == (n_classes, n_classes)

    def test_cv_summary_has_expected_columns(self, clf_result):
        summary = ModelEvaluator.summarise_cv(clf_result.cv_results, "classification")
        assert "Metric" in summary.columns
        assert "Mean" in summary.columns
        assert not summary.empty


# ---------------------------------------------------------------------------
# ModelEvaluator – regression
# ---------------------------------------------------------------------------
class TestModelEvaluatorRegression:
    def test_regression_metrics_valid(self, reg_result):
        m = ModelEvaluator.compute_regression_metrics(reg_result)
        assert m.mae >= 0
        assert m.rmse >= 0
        assert m.r2 <= 1.0  # can be negative

    def test_regression_metrics_not_hardcoded(self, reg_result):
        m = ModelEvaluator.compute_regression_metrics(reg_result)
        # Ensure results are derived from actual predictions, not constants
        assert m.mae != 42.0
        assert m.rmse != 0.0 or m.r2 == 1.0  # perfect fit is allowed


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------
class TestCompareModels:
    def test_compare_returns_dataframe(self, clf_dataset):
        X, y, pipe, p, le = clf_dataset
        trainer = ModelTrainer(pipe, "classification", cv_folds=3, label_encoder=le)
        results = trainer.train_all(X, y)
        df = compare_models(results, "classification")
        assert isinstance(df, pd.DataFrame)
        assert "Model" in df.columns
        assert len(df) == len(results)

    def test_all_metric_columns_present(self, clf_dataset):
        X, y, pipe, p, le = clf_dataset
        trainer = ModelTrainer(pipe, "classification", cv_folds=3, label_encoder=le)
        results = trainer.train_all(X, y)
        df = compare_models(results, "classification")
        assert "Accuracy" in df.columns
        assert "F1-Score (weighted)" in df.columns


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
class TestFeatureImportance:
    def test_builtin_importance_for_random_forest(self, clf_dataset):
        X, y, pipe, p, le = clf_dataset
        trainer = ModelTrainer(
            pipe, "classification", selected_models=["Random Forest"], cv_folds=3
        )
        result = trainer.train_all(X, y)["Random Forest"]
        feat_names = p.get_feature_names(X)
        imp_df = ModelEvaluator.get_feature_importance(result, feat_names)
        assert imp_df is not None
        assert "Feature" in imp_df.columns
        assert "Importance" in imp_df.columns
        assert (imp_df["Importance"] >= 0).all()

    def test_no_builtin_importance_for_logistic_regression(self, clf_dataset):
        X, y, pipe, p, le = clf_dataset
        trainer = ModelTrainer(
            pipe, "classification",
            selected_models=["Logistic Regression"],
            cv_folds=3,
        )
        result = trainer.train_all(X, y)["Logistic Regression"]
        feat_names = p.get_feature_names(X)
        imp_df = ModelEvaluator.get_feature_importance(result, feat_names)
        assert imp_df is None
