"""
Tests for src/preprocessing.py
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.preprocessing import Preprocessor, PreprocessingConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """DataFrame with numeric, categorical, and missing values."""
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame(
        {
            "num_a": rng.normal(0, 1, n),
            "num_b": rng.uniform(10, 100, n),
            "cat_x": rng.choice(["alpha", "beta", "gamma"], n),
            "cat_y": rng.choice(["red", "blue"], n),
            "target": rng.choice(["pos", "neg"], n),
        }
    )
    # Introduce missing values
    df.loc[0:5, "num_a"] = np.nan
    df.loc[10:12, "cat_x"] = np.nan
    return df


@pytest.fixture
def const_df() -> pd.DataFrame:
    """DataFrame with a constant column."""
    return pd.DataFrame(
        {
            "useful": list(range(20)),
            "constant": [5] * 20,
            "target": ["a", "b"] * 10,
        }
    )


# ---------------------------------------------------------------------------
# prepare_features
# ---------------------------------------------------------------------------
class TestPrepareFeatures:
    def test_splits_features_and_target(self, mixed_df):
        p = Preprocessor()
        X, y = p.prepare_features(mixed_df, "target")
        assert "target" not in X.columns
        assert len(y) == len(mixed_df)

    def test_drops_all_nan_columns(self):
        df = pd.DataFrame(
            {"a": [1.0] * 20, "all_nan": [np.nan] * 20, "target": ["x", "y"] * 10}
        )
        p = Preprocessor(PreprocessingConfig(drop_all_nan_cols=True))
        X, _ = p.prepare_features(df, "target")
        assert "all_nan" not in X.columns

    def test_drops_constant_columns(self, const_df):
        p = Preprocessor(PreprocessingConfig(drop_const_cols=True))
        X, _ = p.prepare_features(const_df, "target")
        assert "constant" not in X.columns

    def test_missing_target_raises(self, mixed_df):
        p = Preprocessor()
        with pytest.raises(ValueError, match="Target column"):
            p.prepare_features(mixed_df, "nonexistent")


# ---------------------------------------------------------------------------
# encode_target
# ---------------------------------------------------------------------------
class TestEncodeTarget:
    def test_classification_encoding(self):
        p = Preprocessor()
        y = pd.Series(["cat", "dog", "cat", "dog", "cat"] * 4)
        y_enc, le = p.encode_target(y, "classification")
        assert set(y_enc) == {0, 1}
        assert le is not None
        assert list(le.classes_) == ["cat", "dog"]

    def test_regression_encoding(self):
        p = Preprocessor()
        y = pd.Series([1.5, 2.3, 3.7, 0.1, 5.0] * 4)
        y_enc, le = p.encode_target(y, "regression")
        assert le is None
        assert y_enc.dtype == float

    def test_multiclass_encoding(self):
        p = Preprocessor()
        y = pd.Series(["a", "b", "c", "a", "b", "c"] * 4)
        y_enc, le = p.encode_target(y, "classification")
        assert set(y_enc) == {0, 1, 2}


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------
class TestBuildPipeline:
    def test_pipeline_is_sklearn_pipeline(self, mixed_df):
        p = Preprocessor()
        X, _ = p.prepare_features(mixed_df, "target")
        pipe = p.build_pipeline(X)
        assert isinstance(pipe, Pipeline)

    def test_pipeline_transforms_train_data(self, mixed_df):
        p = Preprocessor()
        X, _ = p.prepare_features(mixed_df, "target")
        pipe = p.build_pipeline(X)
        X_transformed = pipe.fit_transform(X)
        assert X_transformed.shape[0] == len(X)
        assert not np.isnan(X_transformed).any(), "Transformed data should have no NaN"

    def test_standard_scaler_applied(self, mixed_df):
        cfg = PreprocessingConfig(scaler="standard")
        p = Preprocessor(cfg)
        X, _ = p.prepare_features(mixed_df, "target")
        pipe = p.build_pipeline(X)
        X_t = pipe.fit_transform(X)
        # Numeric columns should be roughly zero-mean after standard scaling
        num_col_count = len(p.report_.numeric_columns)
        num_means = X_t[:, :num_col_count].mean(axis=0)
        assert all(abs(m) < 0.5 for m in num_means)

    def test_no_scaler_option(self, mixed_df):
        cfg = PreprocessingConfig(scaler="none")
        p = Preprocessor(cfg)
        X, _ = p.prepare_features(mixed_df, "target")
        pipe = p.build_pipeline(X)
        X_t = pipe.fit_transform(X)
        assert X_t is not None

    def test_minmax_scaler(self, mixed_df):
        cfg = PreprocessingConfig(scaler="minmax")
        p = Preprocessor(cfg)
        X, _ = p.prepare_features(mixed_df, "target")
        pipe = p.build_pipeline(X)
        X_t = pipe.fit_transform(X)
        num_cols = len(p.report_.numeric_columns)
        num_portion = X_t[:, :num_cols]
        assert num_portion.min() >= -0.01  # allow floating point slack
        assert num_portion.max() <= 1.01

    def test_ordinal_encoder(self, mixed_df):
        cfg = PreprocessingConfig(encoder="ordinal")
        p = Preprocessor(cfg)
        X, _ = p.prepare_features(mixed_df, "target")
        pipe = p.build_pipeline(X)
        X_t = pipe.fit_transform(X)
        assert not np.isnan(X_t).any()

    def test_empty_features_raises(self):
        df = pd.DataFrame({"target": ["a"] * 15})
        p = Preprocessor()
        with pytest.raises(ValueError):
            X, _ = p.prepare_features(df, "target")
            p.build_pipeline(X)


# ---------------------------------------------------------------------------
# detect_outliers_iqr
# ---------------------------------------------------------------------------
class TestDetectOutliers:
    def test_detects_extreme_values(self):
        df = pd.DataFrame({"col": [1.0] * 20 + [1000.0]})
        result = Preprocessor.detect_outliers_iqr(df, ["col"])
        assert result["col"] >= 1

    def test_no_outliers_normal_data(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"col": rng.normal(0, 1, 200)})
        result = Preprocessor.detect_outliers_iqr(df, ["col"])
        # Normal data shouldn't have too many outliers
        assert result["col"] < 20  # less than 10%

    def test_missing_column_ignored(self):
        df = pd.DataFrame({"col_a": [1, 2, 3] * 10})
        result = Preprocessor.detect_outliers_iqr(df, ["col_a", "nonexistent"])
        assert "nonexistent" not in result
