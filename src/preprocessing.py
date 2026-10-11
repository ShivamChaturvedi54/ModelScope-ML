"""
ModelScope ML – Preprocessing
Builds sklearn Pipelines with ColumnTransformer so preprocessing is fitted
only on training data, preventing data leakage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class PreprocessingConfig:
    """User-selected preprocessing settings."""

    numeric_imputer: str = "mean"          # "mean" | "median" | "most_frequent"
    categorical_imputer: str = "most_frequent"  # "most_frequent" | "constant"
    scaler: str = "standard"               # "standard" | "minmax" | "none"
    encoder: str = "onehot"               # "onehot" | "ordinal"
    handle_unknown: str = "ignore"        # OHE strategy
    drop_all_nan_cols: bool = True
    drop_const_cols: bool = True


@dataclass
class PreprocessingReport:
    """Summary of what preprocessing was applied."""

    dropped_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    imputed_numeric: bool = False
    imputed_categorical: bool = False
    scaled: bool = False
    encoded: bool = False
    label_encoder: Optional[LabelEncoder] = None
    target_classes: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Core preprocessor
# ---------------------------------------------------------------------------
class Preprocessor:
    """
    Builds and stores a scikit-learn preprocessing pipeline.

    The pipeline is fitted **only** on training data.  Call
    ``build_pipeline`` to create the transformer, then use the returned
    sklearn Pipeline in a combined estimator pipeline.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()
        self.column_transformer_: Optional[ColumnTransformer] = None
        self.report_ = PreprocessingReport()
        self.feature_names_out_: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Split features and target; clean the feature matrix.

        Returns
        -------
        X : pd.DataFrame
        y : pd.Series
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")

        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()

        # Drop fully-NaN columns
        if self.config.drop_all_nan_cols:
            all_nan = X.columns[X.isna().all()].tolist()
            if all_nan:
                X = X.drop(columns=all_nan)
                self.report_.dropped_columns.extend(all_nan)
                logger.info("Dropped all-NaN columns: %s", all_nan)

        # Drop zero-variance columns
        if self.config.drop_const_cols:
            const = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
            if const:
                X = X.drop(columns=const)
                self.report_.dropped_columns.extend(const)
                logger.info("Dropped constant columns: %s", const)

        return X, y

    def encode_target(
        self,
        y: pd.Series,
        task_type: str,
    ) -> tuple[np.ndarray, Optional[LabelEncoder]]:
        """
        Encode the target variable.

        For classification, apply LabelEncoder to ensure integer labels.
        For regression, convert to float.

        Returns
        -------
        y_encoded : np.ndarray
        label_encoder : LabelEncoder | None
        """
        task_type = str(task_type).strip().lower()
        if task_type == "classification":
            le = LabelEncoder()
            y_enc = le.fit_transform(y.astype(str))
            self.report_.label_encoder = le
            self.report_.target_classes = le.classes_.tolist()
            return y_enc, le
        if task_type == "regression":
            return y.astype(float).values, None
        raise ValueError(f"Unsupported task_type '{task_type}'. Expected 'classification' or 'regression'.")

    def build_pipeline(
        self,
        X: pd.DataFrame,
    ) -> Pipeline:
        """
        Build a fitted-ready sklearn Pipeline for the feature matrix X.

        The Pipeline is **not** fitted here; it is returned ready to be
        incorporated into an estimator pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (before splitting).

        Returns
        -------
        sklearn.pipeline.Pipeline
            A Pipeline containing a ``ColumnTransformer`` step.
        """
        cfg = self.config
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        self.report_.numeric_columns = numeric_cols
        self.report_.categorical_columns = categorical_cols

        # --- Numeric pipeline ---
        numeric_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy=cfg.numeric_imputer)),
        ]
        if cfg.scaler != "none":
            scaler = (
                StandardScaler() if cfg.scaler == "standard" else MinMaxScaler()
            )
            numeric_steps.append(("scaler", scaler))
            self.report_.scaled = True

        numeric_pipe = Pipeline(steps=numeric_steps)
        self.report_.imputed_numeric = bool(numeric_cols)

        # --- Categorical pipeline ---
        if cfg.encoder == "onehot":
            encoder = OneHotEncoder(
                handle_unknown=cfg.handle_unknown,
                sparse_output=False,
            )
        else:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )

        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=cfg.categorical_imputer, fill_value="missing")),
                ("encoder", encoder),
            ]
        )
        self.report_.imputed_categorical = bool(categorical_cols)
        self.report_.encoded = bool(categorical_cols)

        # --- ColumnTransformer ---
        transformers: list[tuple] = []
        if numeric_cols:
            transformers.append(("numeric", numeric_pipe, numeric_cols))
        if categorical_cols:
            transformers.append(("categorical", categorical_pipe, categorical_cols))

        if not transformers:
            raise ValueError("No valid feature columns found after cleaning.")

        col_transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )
        self.column_transformer_ = col_transformer

        # Wrap in a Pipeline so it can be stacked with an estimator
        return Pipeline(steps=[("preprocessor", col_transformer)])

    def get_feature_names(self, X: pd.DataFrame) -> list[str]:
        """
        Return feature names after transformation.

        Should be called after the pipeline has been fitted.
        """
        if self.column_transformer_ is None:
            raise RuntimeError("Pipeline has not been built yet.")
        try:
            names = self.column_transformer_.get_feature_names_out()
            return [str(n) for n in names]
        except Exception:
            # Fallback
            return [f"feature_{i}" for i in range(100)]

    @staticmethod
    def detect_outliers_iqr(df: pd.DataFrame, cols: list[str]) -> dict[str, int]:
        """
        Detect outliers using the IQR method.

        Returns
        -------
        dict mapping column name → number of outlier rows
        """
        result: dict[str, int] = {}
        for col in cols:
            if col not in df.columns:
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
            result[col] = n_out
        return result
