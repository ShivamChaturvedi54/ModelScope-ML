"""
ModelScope ML – EDA (Exploratory Data Analysis)
Generates Plotly figures for histograms, box plots, correlation heatmaps,
target distributions, and feature-vs-target relationships.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Colour palette (consistent across charts)
# ---------------------------------------------------------------------------
PALETTE = px.colors.qualitative.Set2
PRIMARY = "#4F46E5"
ACCENT = "#06B6D4"
WARN = "#F59E0B"
DANGER = "#EF4444"
SUCCESS = "#10B981"


# ---------------------------------------------------------------------------
# EDA helper functions
# ---------------------------------------------------------------------------

def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
) -> go.Figure:
    """
    Plot the distribution of the target variable.
    Classification → bar chart of class counts.
    Regression → histogram with KDE-like curve.
    """
    if task_type == "classification":
        counts = df[target_col].value_counts().reset_index()
        counts.columns = [target_col, "count"]
        fig = px.bar(
            counts,
            x=target_col,
            y="count",
            color=target_col,
            color_discrete_sequence=PALETTE,
            title=f"Target Distribution – {target_col}",
            labels={target_col: "Class", "count": "Count"},
        )
        fig.update_layout(showlegend=False)
    else:
        fig = px.histogram(
            df,
            x=target_col,
            nbins=40,
            title=f"Target Distribution – {target_col}",
            color_discrete_sequence=[PRIMARY],
            marginal="box",
        )
    _apply_theme(fig)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, target_col: Optional[str] = None) -> go.Figure:
    """
    Correlation heatmap for numeric columns.
    If target_col is supplied, it is placed last.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough numeric columns for correlation.", showarrow=False)
        return fig

    if target_col and target_col in numeric_df.columns:
        # Reorder so target is last
        cols = [c for c in numeric_df.columns if c != target_col] + [target_col]
        numeric_df = numeric_df[cols]

    corr = numeric_df.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap",
    )
    fig.update_layout(width=700, height=600)
    _apply_theme(fig)
    return fig


def plot_histograms(df: pd.DataFrame, numeric_cols: list[str], n_cols: int = 3) -> go.Figure:
    """
    Small-multiple histograms for each numeric feature.
    """
    cols_to_plot = [c for c in numeric_cols if c in df.columns][:12]  # cap at 12
    if not cols_to_plot:
        fig = go.Figure()
        fig.add_annotation(text="No numeric columns to display.", showarrow=False)
        return fig

    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cols_to_plot)

    for idx, col in enumerate(cols_to_plot):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1
        fig.add_trace(
            go.Histogram(
                x=df[col].dropna(),
                name=col,
                marker_color=PALETTE[idx % len(PALETTE)],
                showlegend=False,
            ),
            row=row,
            col=col_pos,
        )

    fig.update_layout(
        height=280 * n_rows,
        title_text="Feature Distributions",
    )
    _apply_theme(fig)
    return fig


def plot_boxplots(df: pd.DataFrame, numeric_cols: list[str]) -> go.Figure:
    """
    Box plots for all numeric columns side by side.
    """
    cols_to_plot = [c for c in numeric_cols if c in df.columns][:15]
    if not cols_to_plot:
        fig = go.Figure()
        fig.add_annotation(text="No numeric columns to display.", showarrow=False)
        return fig

    fig = go.Figure()
    for idx, col in enumerate(cols_to_plot):
        fig.add_trace(
            go.Box(
                y=df[col].dropna(),
                name=col,
                marker_color=PALETTE[idx % len(PALETTE)],
                boxmean="sd",
            )
        )
    fig.update_layout(
        title="Box Plots – Numeric Features",
        xaxis_title="Feature",
        yaxis_title="Value",
    )
    _apply_theme(fig)
    return fig


def plot_missing_values(df: pd.DataFrame) -> go.Figure:
    """Bar chart of missing-value percentages per column."""
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="✅ No missing values found!",
            showarrow=False,
            font=dict(size=16, color=SUCCESS),
        )
        _apply_theme(fig)
        return fig

    fig = px.bar(
        x=missing.index,
        y=missing.values,
        labels={"x": "Column", "y": "Missing (%)"},
        title="Missing Values per Column",
        color=missing.values,
        color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
    )
    fig.update_layout(coloraxis_showscale=False)
    _apply_theme(fig)
    return fig


def plot_feature_vs_target(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    task_type: str,
) -> go.Figure:
    """
    Plot a single feature against the target.
    - Classification: violin plot coloured by class.
    - Regression: scatter plot with trend line.
    """
    if feature_col not in df.columns or target_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Column not found.", showarrow=False)
        return fig

    if task_type == "classification":
        fig = px.violin(
            df,
            x=target_col,
            y=feature_col if pd.api.types.is_numeric_dtype(df[feature_col]) else None,
            color=target_col,
            box=True,
            color_discrete_sequence=PALETTE,
            title=f"{feature_col} by {target_col}",
        )
    else:
        fig = px.scatter(
            df,
            x=feature_col,
            y=target_col,
            color_discrete_sequence=[PRIMARY],
            title=f"{feature_col} vs {target_col}",
            opacity=0.6,
        )

    _apply_theme(fig)
    return fig


def plot_class_imbalance(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Pie chart showing class distribution for classification tasks."""
    counts = df[target_col].value_counts()
    fig = px.pie(
        names=counts.index,
        values=counts.values,
        title=f"Class Distribution – {target_col}",
        color_discrete_sequence=PALETTE,
        hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    _apply_theme(fig)
    return fig


def plot_pairplot(
    df: pd.DataFrame,
    cols: list[str],
    target_col: Optional[str] = None,
    max_cols: int = 5,
) -> go.Figure:
    """Scatter-matrix (pair plot) for selected numeric columns."""
    cols = cols[:max_cols]
    color_col = target_col if target_col and target_col in df.columns else None
    fig = px.scatter_matrix(
        df,
        dimensions=cols,
        color=color_col,
        color_discrete_sequence=PALETTE,
        title="Scatter Matrix",
        opacity=0.5,
    )
    fig.update_traces(diagonal_visible=False)
    _apply_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_theme(fig: go.Figure) -> None:
    """Apply consistent styling to every figure."""
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
