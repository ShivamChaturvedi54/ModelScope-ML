"""
ModelScope ML – Prediction Engine
Validates inputs, handles unseen categories, and generates predictions
with probabilities / confidence intervals.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class PredictionInput:
    """Raw user-supplied feature values."""

    feature_values: dict[str, Any]


@dataclass
class PredictionOutput:
    """Result of a single prediction."""

    predicted_value: Any
    predicted_label: Optional[str]          # decoded class name or None
    probabilities: Optional[dict[str, float]]  # class → probability
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class PredictionEngine:
    """
    Wraps a fitted sklearn Pipeline to validate and predict from user input.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        feature_names: list[str],
        task_type: str,
        label_encoder=None,
        training_data: Optional[pd.DataFrame] = None,
    ) -> None:
        self.pipeline = pipeline
        self.feature_names = feature_names
        self.task_type = str(task_type).strip().lower()
        self.label_encoder = label_encoder
        self.training_data = training_data  # used for domain validation

        # Pre-compute categorical value sets for validation
        self._cat_value_sets: dict[str, set] = {}
        if training_data is not None:
            for col in feature_names:
                if col in training_data.columns and not pd.api.types.is_numeric_dtype(
                    training_data[col]
                ):
                    self._cat_value_sets[col] = set(training_data[col].dropna().unique())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_input_row(self, user_values: dict[str, Any]) -> pd.DataFrame:
        """
        Convert a dict of user inputs into a one-row DataFrame matching
        the expected feature schema.
        """
        row: dict[str, Any] = {}
        for feat in self.feature_names:
            row[feat] = user_values.get(feat, np.nan)
        return pd.DataFrame([row])

    def predict(self, user_values: dict[str, Any]) -> PredictionOutput:
        """
        Run prediction for a single set of feature values.

        Parameters
        ----------
        user_values : dict[str, Any]
            Feature name → value.

        Returns
        -------
        PredictionOutput
        """
        warnings: list[str] = []
        errors: list[str] = []

        # Validate inputs
        validation_warnings = self._validate_inputs(user_values)
        warnings.extend(validation_warnings)

        # Build input row
        try:
            input_df = self.build_input_row(user_values)
        except Exception as exc:
            errors.append(f"Could not build input: {exc}")
            return PredictionOutput(
                predicted_value=None,
                predicted_label=None,
                probabilities=None,
                warnings=warnings,
                errors=errors,
            )

        # Predict
        try:
            raw_pred = self.pipeline.predict(input_df)[0]
        except Exception as exc:
            errors.append(f"Prediction failed: {exc}")
            return PredictionOutput(
                predicted_value=None,
                predicted_label=None,
                probabilities=None,
                warnings=warnings,
                errors=errors,
            )

        # Decode label
        predicted_label: Optional[str] = None
        if self.task_type == "classification" and self.label_encoder is not None:
            try:
                predicted_label = str(
                    self.label_encoder.inverse_transform([int(raw_pred)])[0]
                )
            except Exception:
                predicted_label = str(raw_pred)
        elif self.task_type == "classification":
            predicted_label = str(raw_pred)

        # Probabilities
        probabilities: Optional[dict[str, float]] = None
        if self.task_type == "classification" and hasattr(self.pipeline, "predict_proba"):
            try:
                proba_arr = self.pipeline.predict_proba(input_df)[0]
                classes = (
                    self.label_encoder.classes_.tolist()
                    if self.label_encoder is not None
                    else [str(i) for i in range(len(proba_arr))]
                )
                probabilities = {
                    cls: round(float(p), 4) for cls, p in zip(classes, proba_arr)
                }
            except Exception as exc:
                logger.warning("Could not compute probabilities: %s", exc)

        return PredictionOutput(
            predicted_value=raw_pred,
            predicted_label=predicted_label,
            probabilities=probabilities,
            warnings=warnings,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _validate_inputs(self, user_values: dict[str, Any]) -> list[str]:
        """Return validation warnings for user inputs."""
        warnings: list[str] = []

        for feat in self.feature_names:
            val = user_values.get(feat)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                warnings.append(
                    f"Feature '{feat}' is missing. The model will use imputation."
                )
                continue

            # Check for unseen categorical values
            if feat in self._cat_value_sets:
                if str(val) not in {str(v) for v in self._cat_value_sets[feat]}:
                    warnings.append(
                        f"Value '{val}' for '{feat}' was not seen during training. "
                        "The model will apply its unknown-handling strategy."
                    )

            # Range check for numeric features
            if self.training_data is not None and feat in self.training_data.columns:
                col_data = self.training_data[feat]
                if pd.api.types.is_numeric_dtype(col_data):
                    try:
                        fval = float(val)
                        q01 = col_data.quantile(0.01)
                        q99 = col_data.quantile(0.99)
                        if fval < q01 or fval > q99:
                            warnings.append(
                                f"Value {fval:.4g} for '{feat}' is outside the 1st–99th "
                                f"percentile range [{q01:.4g}, {q99:.4g}] seen during training."
                            )
                    except (ValueError, TypeError):
                        pass

        return warnings
