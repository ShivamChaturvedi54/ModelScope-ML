"""
ModelScope ML – Model Evaluation
Computes metrics, confusion matrices, ROC curves, and CV summaries.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
)

from .model_trainer import TrainingResult

logger = logging.getLogger(__name__)

PALETTE = px.colors.qualitative.Set2
PRIMARY = "#4F46E5"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    confusion_mat: np.ndarray
    n_classes: int
    classes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "Accuracy": round(self.accuracy, 4),
            "Precision (weighted)": round(self.precision, 4),
            "Recall (weighted)": round(self.recall, 4),
            "F1-Score (weighted)": round(self.f1, 4),
            "ROC-AUC": round(self.roc_auc, 4) if self.roc_auc is not None else "N/A",
        }


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "MAE": round(self.mae, 4),
            "RMSE": round(self.rmse, 4),
            "R²": round(self.r2, 4),
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class ModelEvaluator:
    """
    Computes and visualises model evaluation metrics.

    Works from a ``TrainingResult`` produced by ``ModelTrainer``.
    """

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    @staticmethod
    def compute_classification_metrics(
        result: TrainingResult,
    ) -> ClassificationMetrics:
        y_test = result.y_test
        y_pred = result.y_pred_test
        y_proba = result.y_proba_test
        classes = result.classes or [str(c) for c in np.unique(y_test)]
        n_classes = len(classes)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        roc_auc: Optional[float] = None
        if y_proba is not None:
            try:
                from sklearn.metrics import roc_auc_score
                if n_classes == 2:
                    roc_auc = float(roc_auc_score(y_test, y_proba[:, 1]))
                else:
                    roc_auc = float(
                        roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
                    )
            except Exception as exc:
                logger.warning("ROC-AUC computation failed: %s", exc)

        cm = confusion_matrix(y_test, y_pred)

        return ClassificationMetrics(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            roc_auc=roc_auc,
            confusion_mat=cm,
            n_classes=n_classes,
            classes=classes,
        )

    @staticmethod
    def plot_confusion_matrix(metrics: ClassificationMetrics, model_name: str) -> go.Figure:
        cm = metrics.confusion_mat
        labels = metrics.classes

        fig = px.imshow(
            cm,
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"Confusion Matrix – {model_name}",
            labels=dict(x="Predicted", y="Actual"),
        )
        fig.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            coloraxis_showscale=False,
        )
        _apply_theme(fig)
        return fig

    @staticmethod
    def plot_roc_curve(result: TrainingResult, model_name: str) -> Optional[go.Figure]:
        """Plot ROC curve(s).  Binary → single curve. Multi-class → OvR curves."""
        if result.y_proba_test is None:
            return None

        y_test = result.y_test
        y_proba = result.y_proba_test
        classes = result.classes or [str(c) for c in np.unique(y_test)]
        n_classes = len(classes)

        fig = go.Figure()

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            area = auc(fpr, tpr)
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr, mode="lines", name=f"AUC = {area:.3f}",
                    line=dict(color=PRIMARY, width=2),
                )
            )
        else:
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_test, classes=np.unique(y_test))
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                area = auc(fpr, tpr)
                fig.add_trace(
                    go.Scatter(
                        x=fpr, y=tpr, mode="lines",
                        name=f"{cls} (AUC={area:.3f})",
                        line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                    )
                )

        fig.add_shape(
            type="line", x0=0, y0=0, x1=1, y1=1,
            line=dict(color="#94A3B8", width=1, dash="dash"),
        )
        fig.update_layout(
            title=f"ROC Curve – {model_name}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.02]),
        )
        _apply_theme(fig)
        return fig

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------
    @staticmethod
    def compute_regression_metrics(result: TrainingResult) -> RegressionMetrics:
        y_test = result.y_test
        y_pred = result.y_pred_test

        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = r2_score(y_test, y_pred)

        return RegressionMetrics(mae=float(mae), rmse=rmse, r2=float(r2))

    @staticmethod
    def plot_residuals(result: TrainingResult, model_name: str) -> go.Figure:
        y_test = result.y_test
        y_pred = result.y_pred_test
        residuals = y_test - y_pred

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=y_pred, y=residuals,
                mode="markers",
                marker=dict(color=PRIMARY, opacity=0.6, size=6),
                name="Residuals",
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
        fig.update_layout(
            title=f"Residual Plot – {model_name}",
            xaxis_title="Predicted Values",
            yaxis_title="Residuals (Actual − Predicted)",
        )
        _apply_theme(fig)
        return fig

    @staticmethod
    def plot_actual_vs_predicted(result: TrainingResult, model_name: str) -> go.Figure:
        y_test = result.y_test
        y_pred = result.y_pred_test

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=y_test, y=y_pred,
                mode="markers",
                marker=dict(color=PRIMARY, opacity=0.6, size=6),
                name="Predictions",
            )
        )
        lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        fig.add_trace(
            go.Scatter(
                x=lim, y=lim,
                mode="lines",
                line=dict(color="#94A3B8", dash="dash"),
                name="Perfect Fit",
            )
        )
        fig.update_layout(
            title=f"Actual vs Predicted – {model_name}",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
        )
        _apply_theme(fig)
        return fig

    # ------------------------------------------------------------------
    # Cross-validation summary
    # ------------------------------------------------------------------
    @staticmethod
    def summarise_cv(
        cv_results: dict[str, Any], task_type: str
    ) -> pd.DataFrame:
        """
        Return a tidy DataFrame summarising cross-validation scores.
        """
        rows: list[dict] = []
        key_map = {
            "test_accuracy": "Accuracy",
            "test_f1_weighted": "F1 (weighted)",
            "test_roc_auc_ovr_weighted": "ROC-AUC (OvR)",
            "test_neg_mean_absolute_error": "MAE",
            "test_neg_root_mean_squared_error": "RMSE",
            "test_r2": "R²",
        }
        for raw_key, label in key_map.items():
            if raw_key in cv_results:
                vals = np.array(cv_results[raw_key])
                if "neg_" in raw_key:
                    vals = np.abs(vals)
                rows.append(
                    {
                        "Metric": label,
                        "Mean": round(float(vals.mean()), 4),
                        "Std": round(float(vals.std()), 4),
                        "Min": round(float(vals.min()), 4),
                        "Max": round(float(vals.max()), 4),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def plot_cv_scores(
        cv_results: dict[str, Any], model_name: str, task_type: str
    ) -> go.Figure:
        """Box/strip plot of CV fold scores per metric."""
        key_map = {
            "test_accuracy": "Accuracy",
            "test_f1_weighted": "F1",
            "test_r2": "R²",
        }
        traces = []
        for raw_key, label in key_map.items():
            if raw_key not in cv_results:
                continue
            vals = np.array(cv_results[raw_key])
            if "neg_" in raw_key:
                vals = np.abs(vals)
            traces.append(go.Box(y=vals.tolist(), name=label, boxpoints="all", jitter=0.4))

        fig = go.Figure(traces)
        fig.update_layout(
            title=f"Cross-Validation Scores – {model_name}",
            yaxis_title="Score",
        )
        _apply_theme(fig)
        return fig

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    @staticmethod
    def get_feature_importance(
        result: TrainingResult,
        feature_names: list[str],
    ) -> Optional[pd.DataFrame]:
        """
        Extract built-in feature importance from tree models.

        Returns
        -------
        pd.DataFrame with columns ['Feature', 'Importance'] or None.
        """
        estimator = result.pipeline.named_steps.get("estimator")
        if estimator is None:
            return None
        if not hasattr(estimator, "feature_importances_"):
            return None

        importances = estimator.feature_importances_
        n = min(len(importances), len(feature_names))
        df = pd.DataFrame(
            {"Feature": feature_names[:n], "Importance": importances[:n]}
        ).sort_values("Importance", ascending=False)
        return df

    @staticmethod
    def get_permutation_importance(
        result: TrainingResult,
        feature_names: list[str],
        n_repeats: int = 10,
    ) -> Optional[pd.DataFrame]:
        """
        Compute model-agnostic permutation importance on the test set.
        """
        try:
            perm = permutation_importance(
                result.pipeline,
                result.X_test,
                result.y_test,
                n_repeats=n_repeats,
                random_state=42,
                n_jobs=-1,
            )
        except Exception as exc:
            logger.warning("Permutation importance failed: %s", exc)
            return None

        n = min(len(perm.importances_mean), len(feature_names))
        df = pd.DataFrame(
            {
                "Feature": feature_names[:n],
                "Importance": perm.importances_mean[:n],
                "Std": perm.importances_std[:n],
            }
        ).sort_values("Importance", ascending=False)
        return df

    @staticmethod
    def plot_feature_importance(df: pd.DataFrame, title: str, top_n: int = 20) -> go.Figure:
        """Horizontal bar chart of feature importances."""
        df_plot = df.head(top_n).sort_values("Importance")
        has_std = "Std" in df_plot.columns

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df_plot["Importance"],
                y=df_plot["Feature"],
                orientation="h",
                marker_color=PRIMARY,
                error_x=dict(
                    type="data", array=df_plot["Std"].tolist(), visible=True
                ) if has_std else None,
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=max(350, top_n * 25),
        )
        _apply_theme(fig)
        return fig


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
def compare_models(
    results: dict[str, TrainingResult],
    task_type: str,
) -> pd.DataFrame:
    """
    Build a comparison DataFrame for all trained models.
    """
    rows = []
    for name, result in results.items():
        if task_type == "classification":
            m = ModelEvaluator.compute_classification_metrics(result)
            row = {"Model": name, **m.to_dict()}
        else:
            m = ModelEvaluator.compute_regression_metrics(result)
            row = {"Model": name, **m.to_dict()}
        rows.append(row)
    return pd.DataFrame(rows)


def plot_model_comparison(comparison_df: pd.DataFrame, task_type: str) -> go.Figure:
    """Grouped bar chart comparing models across metrics."""
    metrics = [c for c in comparison_df.columns if c != "Model"]
    # Filter out N/A values for numeric comparison
    numeric_df = comparison_df.copy()
    for m in metrics:
        numeric_df[m] = pd.to_numeric(numeric_df[m], errors="coerce")

    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(
            go.Bar(
                name=metric,
                x=numeric_df["Model"],
                y=numeric_df[metric],
                text=numeric_df[metric].round(4),
                textposition="outside",
            )
        )
    fig.update_layout(
        barmode="group",
        title="Model Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.15]) if task_type == "classification" else {},
    )
    _apply_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _apply_theme(fig: go.Figure) -> None:
    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=13, color="#1E293B"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        margin=dict(t=50, b=40, l=40, r=20),
        title_font=dict(size=16, color="#1E293B"),
        hoverlabel=dict(bgcolor="#FFFFFF", font_size=12),
    )
    fig.update_xaxes(gridcolor="#E2E8F0", linecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#E2E8F0", linecolor="#CBD5E1")
