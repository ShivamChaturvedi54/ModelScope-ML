"""
ModelScope ML – Model Trainer
Trains classification and regression models using sklearn Pipelines with
reproducible random seeds and proper train/test splits.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

CLASSIFICATION_MODELS: dict[str, Any] = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    ),
}

REGRESSION_MODELS: dict[str, Any] = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(
        max_depth=6, random_state=RANDOM_STATE
    ),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    ),
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class TrainingResult:
    """Complete training result for one model."""

    model_name: str
    pipeline: Pipeline          # fitted full pipeline (preprocessor + estimator)
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    y_pred_train: np.ndarray
    y_pred_test: np.ndarray
    y_proba_test: Optional[np.ndarray]  # None for regression / unsupported
    cv_results: dict[str, Any]
    task_type: str
    feature_names_in: list[str]
    classes: Optional[list[str]] = None   # classification class labels
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class ModelTrainer:
    """
    Trains one or more models using full sklearn Pipelines.

    Usage
    -----
    trainer = ModelTrainer(preprocessor_pipeline, task_type="classification")
    results = trainer.train_all(X, y)
    """

    def __init__(
        self,
        preprocessor_pipeline: Pipeline,
        task_type: str,
        selected_models: Optional[list[str]] = None,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
        cv_folds: int = CV_FOLDS,
        label_encoder=None,
    ) -> None:
        task_type = str(task_type).strip().lower()
        if task_type not in ("classification", "regression"):
            raise ValueError(f"task_type must be 'classification' or 'regression', got '{task_type}'")

        self.preprocessor_pipeline = preprocessor_pipeline
        self.task_type = task_type
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.label_encoder = label_encoder

        model_registry = (
            CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
        )
        if selected_models:
            unknown = set(selected_models) - set(model_registry)
            if unknown:
                raise ValueError(f"Unknown models: {unknown}")
            self.models = {k: v for k, v in model_registry.items() if k in selected_models}
        else:
            self.models = dict(model_registry)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def split_data(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Perform a stratified (classification) or regular (regression) train/test split.
        """
        stratify = y if self.task_type == "classification" else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify,
            )
        except ValueError:
            # Stratification can fail when classes have very few samples
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
            )
        return X_train, X_test, y_train, y_test

    def train_all(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> dict[str, TrainingResult]:
        """
        Train all selected models and return a dict of TrainingResult.

        Cross-validation is performed on the training set only.
        Final evaluation data (X_test, y_test) is returned for held-out testing.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (pre-split).
        y : np.ndarray
            Encoded target array.

        Returns
        -------
        dict[str, TrainingResult]
        """
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        results: dict[str, TrainingResult] = {}

        for name, estimator in self.models.items():
            logger.info("Training %s …", name)
            result = self._train_single(
                name=name,
                estimator=estimator,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                X_full=X,
                y_full=y,
            )
            results[name] = result

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _train_single(
        self,
        *,
        name: str,
        estimator: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        X_full: pd.DataFrame,
        y_full: np.ndarray,
    ) -> TrainingResult:
        """Build full pipeline, run CV on train set, then fit on full train."""
        # Build the full pipeline: preprocessor + estimator
        full_pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor_pipeline.named_steps["preprocessor"]),
                ("estimator", estimator),
            ]
        )

        # Cross-validation on training set
        cv_results = self._cross_validate(full_pipeline, X_train, y_train)

        # Fit on full training data
        full_pipeline.fit(X_train, y_train)

        # Predictions
        y_pred_train = full_pipeline.predict(X_train)
        y_pred_test = full_pipeline.predict(X_test)

        # Probabilities (classification only)
        y_proba_test: Optional[np.ndarray] = None
        if self.task_type == "classification" and hasattr(full_pipeline, "predict_proba"):
            try:
                y_proba_test = full_pipeline.predict_proba(X_test)
            except Exception:
                pass

        # Warnings
        warnings: list[str] = []
        if self.task_type == "classification":
            classes, counts = np.unique(y_train, return_counts=True)
            if counts.min() / counts.max() < 0.2:
                warnings.append(
                    "Class imbalance detected in training data. "
                    "Accuracy alone may be misleading."
                )

        classes_labels = None
        if self.label_encoder is not None:
            classes_labels = self.label_encoder.classes_.tolist()

        return TrainingResult(
            model_name=name,
            pipeline=full_pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            y_proba_test=y_proba_test,
            cv_results=cv_results,
            task_type=self.task_type,
            feature_names_in=X_train.columns.tolist(),
            classes=classes_labels,
            warnings=warnings,
        )

    def _cross_validate(
        self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: np.ndarray
    ) -> dict[str, Any]:
        """Run k-fold cross-validation on the training set."""
        if self.task_type == "classification":
            scoring = ["accuracy", "f1_weighted", "roc_auc_ovr_weighted"]
            try:
                cv = StratifiedKFold(
                    n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
                )
            except Exception:
                cv = KFold(
                    n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
                )
        else:
            scoring = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]
            cv = KFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )

        try:
            raw = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
                error_score="raise",
            )
        except Exception as exc:
            logger.warning("CV failed with full scoring, falling back: %s", exc)
            # Fallback to single metric
            fallback = "accuracy" if self.task_type == "classification" else "r2"
            raw = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=self.cv_folds,
                scoring=fallback,
                return_train_score=False,
            )

        # Convert arrays to lists for serialisability
        return {k: v.tolist() for k, v in raw.items()}
