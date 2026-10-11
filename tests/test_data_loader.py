"""
Tests for src/data_loader.py
"""
import io
import pytest
import pandas as pd
import numpy as np

from src.data_loader import DataLoader, SAMPLE_DATASETS, ValidationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_clf_df() -> pd.DataFrame:
    """Small classification-ready dataframe."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, 50),
            "feature_b": rng.normal(5, 2, 50),
            "category": rng.choice(["cat", "dog"], 50),
            "label": rng.choice(["yes", "no"], 50),
        }
    )


@pytest.fixture
def simple_reg_df() -> pd.DataFrame:
    """Small regression-ready dataframe."""
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 80)
    return pd.DataFrame({"x1": x, "x2": rng.normal(0, 1, 80), "target": 2 * x + rng.normal(0, 0.1, 80)})


# ---------------------------------------------------------------------------
# load_sample_dataset
# ---------------------------------------------------------------------------
class TestLoadSampleDataset:
    def test_iris_loads(self):
        df = DataLoader.load_sample_dataset("Iris (Classification)")
        assert "target" in df.columns
        assert len(df) == 150
        assert df["target"].nunique() == 3

    def test_breast_cancer_loads(self):
        df = DataLoader.load_sample_dataset("Breast Cancer Wisconsin (Classification)")
        assert "target" in df.columns
        assert len(df) == 569

    def test_wine_loads(self):
        df = DataLoader.load_sample_dataset("Wine Quality (Classification)")
        assert "target" in df.columns
        assert len(df) == 178

    def test_diabetes_loads(self):
        df = DataLoader.load_sample_dataset("Diabetes (Regression)")
        assert "target" in df.columns
        assert pd.api.types.is_numeric_dtype(df["target"])
        assert len(df) == 442

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            DataLoader.load_sample_dataset("NonExistentDataset")


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------
class TestLoadCSV:
    def test_valid_csv(self, simple_clf_df):
        buf = io.BytesIO(simple_clf_df.to_csv(index=False).encode())
        loaded = DataLoader.load_csv(buf)
        assert loaded.shape == simple_clf_df.shape

    def test_empty_csv_raises(self):
        buf = io.BytesIO(b"col_a,col_b\n")  # header only, no rows
        with pytest.raises(ValueError):
            DataLoader.load_csv(buf)

    def test_malformed_csv_raises(self):
        buf = io.BytesIO(b"\x00\x01\x02\x03invalid bytes\x00")
        with pytest.raises((ValueError, Exception)):
            DataLoader.load_csv(buf)

    def test_csv_with_missing_values(self):
        csv = "a,b,c\n1,,3\n4,5,\n7,8,9\n1,2,3\n5,6,7\n8,9,1\n2,3,4\n5,6,7\n8,9,1\n2,3,4\n"
        buf = io.BytesIO(csv.encode())
        df = DataLoader.load_csv(buf)
        assert df.isna().sum().sum() > 0


# ---------------------------------------------------------------------------
# validate_dataset
# ---------------------------------------------------------------------------
class TestValidateDataset:
    def test_valid_classification(self, simple_clf_df):
        result = DataLoader.validate_dataset(simple_clf_df, "label", "classification")
        assert result.is_valid
        assert len(result.errors) == 0

    def test_valid_regression(self, simple_reg_df):
        result = DataLoader.validate_dataset(simple_reg_df, "target", "regression")
        assert result.is_valid

    def test_too_few_rows(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": ["x", "y"]})
        result = DataLoader.validate_dataset(df, "c", "classification")
        assert not result.is_valid
        assert any("rows" in e for e in result.errors)

    def test_missing_target_column(self, simple_clf_df):
        result = DataLoader.validate_dataset(simple_clf_df, "nonexistent_col", "classification")
        assert not result.is_valid
        assert any("not found" in e for e in result.errors)

    def test_regression_with_categorical_target(self, simple_clf_df):
        result = DataLoader.validate_dataset(simple_clf_df, "label", "regression")
        assert not result.is_valid
        assert any("not numeric" in e for e in result.errors)

    def test_constant_column_warning(self, simple_clf_df):
        df = simple_clf_df.copy()
        df["constant_col"] = 42
        result = DataLoader.validate_dataset(df, "label", "classification")
        assert any("constant" in w.lower() or "zero variance" in w.lower() for w in result.warnings)

    def test_high_cardinality_classification_warning(self):
        df = pd.DataFrame(
            {"feature": list(range(100)), "target": list(range(100))}
        )
        result = DataLoader.validate_dataset(df, "target", "classification")
        assert any("unique values" in w or "high" in w for w in result.warnings)

    def test_id_column_leakage_warning(self, simple_clf_df):
        df = simple_clf_df.copy()
        df["row_id"] = range(len(df))
        result = DataLoader.validate_dataset(df, "label", "classification")
        assert any("identifier" in w.lower() or "leakage" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# get_dataset_info
# ---------------------------------------------------------------------------
class TestGetDatasetInfo:
    def test_info_correctness(self, simple_clf_df):
        info = DataLoader.get_dataset_info(simple_clf_df, "label", "classification")
        assert info.n_rows == 50
        assert info.n_cols == 4
        assert "label" in info.column_dtypes
        assert "feature_a" in info.numeric_cols
        assert "category" in info.categorical_cols

    def test_missing_values_counted(self):
        df = pd.DataFrame(
            {
                "a": [1.0, None, 3.0] + [float(i) for i in range(10)],
                "b": ["x", "y", None] + ["z"] * 10,
                "target": ["cls1", "cls2"] * 6 + ["cls1"],
            }
        )
        info = DataLoader.get_dataset_info(df, "target", "classification")
        assert info.missing_counts["a"] == 1
        assert info.missing_counts["b"] == 1

    def test_duplicate_rows_counted(self, simple_clf_df):
        df = pd.concat([simple_clf_df, simple_clf_df.iloc[:5]], ignore_index=True)
        info = DataLoader.get_dataset_info(df)
        assert info.duplicate_rows == 5
