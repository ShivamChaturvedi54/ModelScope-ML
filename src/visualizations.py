"""
ModelScope ML – Visualizations
Shared chart helpers used across the UI pages.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = px.colors.qualitative.Set2
PRIMARY = "#4F46E5"
ACCENT = "#06B6D4"


def kpi_card_html(label: str, value: str, subtitle: str = "", color: str = PRIMARY) -> str:
    """Return an HTML string for a styled KPI card (used in st.markdown)."""
    return f"""
<div style="
    background: linear-gradient(135deg, {color}18 0%, {color}05 100%);
    border: 1px solid {color}30;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    font-family: Inter, sans-serif;
">
    <div style="font-size: 0.78rem; color: #64748B; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;">{label}</div>
    <div style="font-size: 2rem; font-weight: 700; color: {color}; margin: 6px 0;">{value}</div>
    <div style="font-size: 0.75rem; color: #94A3B8;">{subtitle}</div>
</div>
"""


def badge_html(text: str, color: str = "#4F46E5") -> str:
    """Inline badge HTML."""
    return (
        f'<span style="background:{color}20;color:{color};border:1px solid {color}40;'
        f'border-radius:999px;padding:2px 10px;font-size:0.78rem;font-weight:600;">{text}</span>'
    )


def plot_training_history_placeholder() -> go.Figure:
    """Return a placeholder figure when no model is trained yet."""
    fig = go.Figure()
    fig.add_annotation(
        text="Train a model to see results here.",
        showarrow=False,
        font=dict(size=14, color="#94A3B8"),
    )
    _apply_theme(fig)
    return fig


def plot_probability_bar(probabilities: dict[str, float]) -> go.Figure:
    """Bar chart for classification probabilities."""
    classes = list(probabilities.keys())
    values = list(probabilities.values())

    colors = [PRIMARY if v == max(values) else ACCENT for v in values]

    fig = go.Figure(
        go.Bar(
            x=classes,
            y=values,
            marker_color=colors,
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Class",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1.15]),
    )
    _apply_theme(fig)
    return fig


def plot_gauge(value: float, title: str, min_val: float = 0.0, max_val: float = 1.0) -> go.Figure:
    """Gauge chart for a single metric."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [min_val, max_val]},
                "bar": {"color": PRIMARY},
                "steps": [
                    {"range": [min_val, max_val * 0.5], "color": "#FEF3C7"},
                    {"range": [max_val * 0.5, max_val * 0.75], "color": "#D1FAE5"},
                    {"range": [max_val * 0.75, max_val], "color": "#DBEAFE"},
                ],
            },
            number={"valueformat": ".3f"},
        )
    )
    fig.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
    _apply_theme(fig)
    return fig


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
