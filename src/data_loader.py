"""
ModelScope ML – Data Loader
Handles CSV uploads, built-in sample datasets, and dataset validation.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
)

logger = logging.getLogger(__name__)


def _normalize_task_type(task_type: Optional[str]) -> Optional[str]:
    """Normalize task names while tolerating case and whitespace differences."""
    if task_type is None:
        return None
    return str(task_type).strip().lower()


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
SAMPLE_DATASETS: dict[str, str] = {
    "Iris (Classification)": "iris",
    "Breast Cancer Wisconsin (Classification)": "breast_cancer",
    "Wine Quality (Classification)": "wine",
    "Diabetes (Regression)": "diabetes",
}

MAX_UPLOAD_SIZE_MB = 50
MIN_ROWS = 10
MIN_COLS = 2


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DatasetInfo:
    """Summary statistics about a loaded dataset."""

    n_rows: int
    n_cols: int
    column_dtypes: dict[str, str]
    missing_counts: dict[str, int]
    missing_pct: dict[str, float]
    duplicate_rows: int
    numeric_cols: list[str]
    categorical_cols: list[str]
    target_col: Optional[str] = None
    task_type: Optional[str] = None  # "classification" | "regression"
    warnings: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Outcome of dataset validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------
class DataLoader:
    """
    Loads datasets from various sources and performs validation.

    Supports:
    - Built-in sklearn sample datasets
    - User-uploaded CSV files
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @staticmethod
    def load_sample_dataset(name: str) -> pd.DataFrame:
        """
        Load a built-in sample dataset by key name.

        Parameters
        ----------
        name : str
            One of the keys in ``SAMPLE_DATASETS``.

        Returns
        -------
        pd.DataFrame
            DataFrame including all features **and** the target column.

        Raises
        ------
        ValueError
            If the dataset name is not recognized.
        """
        loaders = {
            "iris": DataLoader._load_iris,
            "breast_cancer": DataLoader._load_breast_cancer,
            "wine": DataLoader._load_wine,
            "diabetes": DataLoader._load_diabetes,
        }

        alias_map = {
            "iris": "iris",
            "iris (classification)": "iris",
            "iris classification": "iris",
            "breast cancer wisconsin": "breast_cancer",
            "breast cancer wisconsin (classification)": "breast_cancer",
            "breast cancer wisconsin classification": "breast_cancer",
            "wine quality": "wine",
            "wine quality (classification)": "wine",
            "wine quality classification": "wine",
            "diabetes": "diabetes",
            "diabetes (regression)": "diabetes",
            "diabetes regression": "diabetes",
        }

        normalized = str(name).strip().lower()
        key = SAMPLE_DATASETS.get(name, name)
        if normalized in alias_map:
            key = alias_map[normalized]
        else:
            key = SAMPLE_DATASETS.get(name, normalized)
            key = key.strip().lower() if isinstance(key, str) else key

        if key not in loaders:
            raise ValueError(
                f"Unknown dataset '{name}'. Available: {list(SAMPLE_DATASETS.keys())}"
            )
        return loaders[key]()

    @staticmethod
    def load_csv(file_obj: io.BytesIO | str, *, encoding: str = "utf-8") -> pd.DataFrame:
        """
        Load a CSV from a file-like object or path.

        Parameters
        ----------
        file_obj : BytesIO | str
            Uploaded file buffer or filesystem path.
        encoding : str
            Character encoding (default: utf-8).

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        ValueError
            If the file is empty or cannot be parsed.
        """
        try:
            df = pd.read_csv(file_obj, encoding=encoding)
        except UnicodeDecodeError:
            # Fallback for common Windows encodings
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            df = pd.read_csv(file_obj, encoding="latin-1")
        except (pd.errors.EmptyDataError, ValueError, TypeError, OSError) as exc:
            raise ValueError(f"Could not read CSV file: {exc}") from exc

        if df.empty:
            raise ValueError("The uploaded file contains no data rows.")

        return df

    @staticmethod
    def validate_dataset(
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate a dataset for ML suitability.

        Checks minimum size, column types, target validity, and potential
        data-leakage columns.

        Parameters
        ----------
        df : pd.DataFrame
        target_col : str, optional
        task_type : str, optional
            "classification" or "regression"

        Returns
        -------
        ValidationResult
        """
        task_type = _normalize_task_type(task_type)
        errors: list[str] = []
        warnings: list[str] = []

        # --- Minimum size ---
        if len(df) < MIN_ROWS:
            errors.append(
                f"Dataset has only {len(df)} rows. At least {MIN_ROWS} rows are required."
            )
        if len(df.columns) < MIN_COLS:
            errors.append(
                f"Dataset has only {len(df.columns)} column(s). "
                f"At least {MIN_COLS} columns are required."
            )

        # --- All-NaN columns ---
        all_nan = df.columns[df.isna().all()].tolist()
        if all_nan:
            warnings.append(
                f"Column(s) {all_nan} are entirely missing and will be dropped."
            )

        # --- Constant columns ---
        const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if const_cols:
            warnings.append(
                f"Column(s) {const_cols} have zero variance (constant). "
                "They will not help the model."
            )

        # --- Target column validation ---
        if target_col:
            if target_col not in df.columns:
                errors.append(f"Target column '{target_col}' not found in dataset.")
            else:
                n_unique = df[target_col].nunique()

                # High-cardinality check for classification
                if task_type == "classification":
                    if n_unique > 50:
                        warnings.append(
                            f"Target '{target_col}' has {n_unique} unique values, "
                            "which is high for classification. "
                            "Consider using regression or reducing classes."
                        )
                    if n_unique < 2:
                        errors.append(
                            f"Target '{target_col}' has only one unique value. "
                            "Classification requires at least 2 classes."
                        )
                    # Detect potential ID / leakage columns
                    if n_unique == len(df):
                        warnings.append(
                            f"Target '{target_col}' has a unique value per row, "
                            "which suggests it may be an identifier, not a label."
                        )

                # Regression target check
                if task_type == "regression":
                    if not pd.api.types.is_numeric_dtype(df[target_col]):
                        errors.append(
                            f"Target '{target_col}' is not numeric, "
                            "which is required for regression."
                        )

                # Leakage heuristic: column name hints
                leakage_keywords = ["id", "index", "uuid", "key", "row_num", "record_id"]
                for col in df.columns:
                    if col == target_col:
                        continue
                    if any(kw in col.lower() for kw in leakage_keywords):
                        warnings.append(
                            f"Column '{col}' may be an identifier. "
                            "Consider dropping it to avoid data leakage."
                        )

        # --- High missing-value columns ---
        missing_pct = df.isna().mean()
        high_missing = missing_pct[missing_pct > 0.5].index.tolist()
        if high_missing:
            warnings.append(
                f"Column(s) {high_missing} have >50 % missing values. "
                "Consider dropping them."
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    @staticmethod
    def get_dataset_info(
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> DatasetInfo:
        """
        Compute summary statistics for a dataset.

        Parameters
        ----------
        df : pd.DataFrame
        target_col : str, optional
        task_type : str, optional

        Returns
        -------
        DatasetInfo
        """
        task_type = _normalize_task_type(task_type)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        missing_counts = df.isna().sum().to_dict()
        n_rows = len(df)
        missing_pct = {
            col: round(cnt / n_rows * 100, 2) if n_rows else 0.0
            for col, cnt in missing_counts.items()
        }

        # Build warnings
        val = DataLoader.validate_dataset(df, target_col, task_type)
        warnings = val.warnings + val.errors  # surface all notes

        return DatasetInfo(
            n_rows=n_rows,
            n_cols=len(df.columns),
            column_dtypes={col: str(df[col].dtype) for col in df.columns},
            missing_counts=missing_counts,
            missing_pct=missing_pct,
            duplicate_rows=int(df.duplicated().sum()),
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            target_col=target_col,
            task_type=task_type,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Private helpers – sample datasets
    # ------------------------------------------------------------------
    @staticmethod
    def _load_iris() -> pd.DataFrame:
        bunch = load_iris(as_frame=True)
        df = bunch.frame.copy()
        df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
        df["target"] = bunch.target.map(dict(enumerate(bunch.target_names)))
        df = df.drop(columns=["target_x"] if "target_x" in df.columns else [])
        # sklearn frame already has 'target' – rename sklearn's numeric target
        if "target" in df.columns and df["target"].dtype in [np.int64, np.int32]:
            df["species"] = df["target"].map(dict(enumerate(bunch.target_names)))
            df = df.drop(columns=["target"])
            df = df.rename(columns={"species": "target"})
        return df

    @staticmethod
    def _load_breast_cancer() -> pd.DataFrame:
        bunch = load_breast_cancer(as_frame=True)
        df = bunch.frame.copy()
        # 'target' is 0/1; add readable label
        df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})
        df = df.drop(columns=["target"])
        df = df.rename(columns={"diagnosis": "target"})
        return df

    @staticmethod
    def _load_wine() -> pd.DataFrame:
        bunch = load_wine(as_frame=True)
        df = bunch.frame.copy()
        df["wine_class"] = df["target"].map(
            {i: f"class_{i}" for i in range(3)}
        )
        df = df.drop(columns=["target"])
        df = df.rename(columns={"wine_class": "target"})
        return df

    @staticmethod
    def _load_diabetes() -> pd.DataFrame:
        bunch = load_diabetes(as_frame=True)
        df = bunch.frame.copy()
        # 'target' column already present (continuous)
        return df
