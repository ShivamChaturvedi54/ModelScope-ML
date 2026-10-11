"""
ModelScope ML – Main Streamlit Application
A production-quality Machine Learning Prediction and Model Evaluation Platform.
"""
from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Internal modules
from src.data_loader import DataLoader, SAMPLE_DATASETS, DatasetInfo
from src.eda import (
    plot_boxplots,
    plot_class_imbalance,
    plot_correlation_heatmap,
    plot_feature_vs_target,
    plot_histograms,
    plot_missing_values,
    plot_pairplot,
    plot_target_distribution,
)
from src.evaluation import (
    ModelEvaluator,
    compare_models,
    plot_model_comparison,
)
from src.model_trainer import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    ModelTrainer,
    TrainingResult,
)
from src.prediction import PredictionEngine
from src.preprocessing import PreprocessingConfig, Preprocessor
from src.visualizations import (
    kpi_card_html,
    plot_probability_bar,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ModelScope ML",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Main container */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {
    color: #94A3B8 !important;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Section headers */
.section-header {
    border-left: 4px solid #4F46E5;
    padding-left: 12px;
    margin: 1.5rem 0 1rem;
    font-size: 1.15rem;
    font-weight: 700;
    color: #1E293B;
}

/* Alert boxes */
.info-box {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.875rem;
    color: #1D4ED8;
    margin: 0.5rem 0;
}
.warn-box {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.875rem;
    color: #92400E;
    margin: 0.5rem 0;
}
.success-box {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.875rem;
    color: #166534;
    margin: 0.5rem 0;
}

/* Metric cards row */
.metric-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 1rem 0;
}

/* Table styling */
.dataframe {
    font-size: 0.85rem !important;
}

/* Divider */
.styled-divider {
    border: none;
    border-top: 2px solid #E2E8F0;
    margin: 1.5rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults: dict[str, Any] = {
        "raw_df": None,
        "raw_dataset_name": None,
        "df": None,
        "dataset_name": None,
        "target_col": None,
        "task_type": "classification",
        "info": None,
        "preprocessor": None,
        "X": None,
        "y": None,
        "label_encoder": None,
        "training_results": {},
        "selected_model": None,
        "page": "Overview",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = [
    "🏠 Overview",
    "📂 Dataset Explorer",
    "🔬 Data Cleaning & EDA",
    "🤖 Model Training",
    "📊 Model Comparison",
    "🎯 Prediction Playground",
    "🧠 Model Insights",
]

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:20px 0 10px;">
            <div style="font-size:2rem;">🔬</div>
            <div style="font-size:1.25rem;font-weight:700;color:#E2E8F0;">ModelScope ML</div>
            <div style="font-size:0.72rem;color:#64748B;margin-top:2px;">ML Prediction Platform</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    page_key = page.split(" ", 1)[1]  # strip emoji

    st.divider()

    # Dataset status indicator
    if st.session_state.df is not None:
        df_info = st.session_state.df
        st.markdown(
            f"""
            <div style="font-size:0.72rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;">Active Dataset</div>
            <div style="color:#10B981;font-weight:600;font-size:0.9rem;">{st.session_state.dataset_name or 'Uploaded CSV'}</div>
            <div style="color:#64748B;font-size:0.78rem;">{df_info.shape[0]:,} rows × {df_info.shape[1]:,} cols</div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.target_col:
            st.markdown(
                f"""
                <div style="margin-top:8px;font-size:0.72rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;">Target</div>
                <div style="color:#F59E0B;font-weight:600;font-size:0.85rem;">{st.session_state.target_col}</div>
                <div style="color:#64748B;font-size:0.78rem;">{st.session_state.task_type.title()}</div>
                """,
                unsafe_allow_html=True,
            )
        if st.session_state.training_results:
            st.markdown(
                f"""
                <div style="margin-top:8px;font-size:0.72rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;">Models Trained</div>
                <div style="color:#4F46E5;font-weight:600;font-size:0.85rem;">{len(st.session_state.training_results)}</div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="color:#475569;font-size:0.82rem;">No dataset loaded yet.</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        '<div style="color:#475569;font-size:0.72rem;">Built with Streamlit + Scikit-learn</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# Page: Overview
# ===========================================================================
def page_overview() -> None:
    st.markdown("## 🏠 Welcome to **ModelScope ML**")
    st.markdown(
        "A production-quality machine learning platform for training, evaluating, "
        "and deploying classification and regression models – all in your browser."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            kpi_card_html("Models Supported", "6", "3 classification + 3 regression"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            kpi_card_html("Built-in Datasets", "4", "Iris, Cancer, Wine, Diabetes", color="#06B6D4"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            kpi_card_html("EDA Charts", "8+", "Interactive Plotly visuals", color="#10B981"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            kpi_card_html("Zero Setup", "100%", "No API keys needed", color="#F59E0B"),
            unsafe_allow_html=True,
        )

    st.divider()

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("### 🚀 Quick Start")
        st.markdown(
            """
1. Go to **📂 Dataset Explorer** to load a sample or upload your own CSV
2. Select the **target column** and task type (classification / regression)
3. Explore data in **🔬 Data Cleaning & EDA**
4. Train models in **🤖 Model Training**
5. Compare models in **📊 Model Comparison**
6. Make predictions in **🎯 Prediction Playground**
7. Inspect feature importance in **🧠 Model Insights**
            """
        )

    with col_b:
        st.markdown("### 🧩 Supported Algorithms")
        st.markdown("**Classification**")
        for m in CLASSIFICATION_MODELS:
            st.markdown(f"- {m}")
        st.markdown("**Regression**")
        for m in REGRESSION_MODELS:
            st.markdown(f"- {m}")

    st.divider()
    st.markdown("### 📐 Evaluation Methodology")
    tab1, tab2 = st.tabs(["Classification Metrics", "Regression Metrics"])
    with tab1:
        st.markdown(
            """
| Metric | Description |
|---|---|
| **Accuracy** | Fraction of correctly classified samples. Simple but misleading on imbalanced data. |
| **Precision** | Of all predicted positives, how many are truly positive? |
| **Recall** | Of all actual positives, how many did we catch? |
| **F1-Score** | Harmonic mean of Precision and Recall. Balanced metric for imbalanced classes. |
| **ROC-AUC** | Area under the ROC curve. Measures separability across all thresholds (binary & OvR multi-class). |
| **Confusion Matrix** | Per-class true/false positive and negative counts. |
            """
        )
    with tab2:
        st.markdown(
            """
| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error – average of absolute differences. Robust to outliers. |
| **RMSE** | Root Mean Squared Error – penalises large errors more heavily. |
| **R²** | Coefficient of determination. 1.0 = perfect fit; 0 = predicts the mean; negative = worse than baseline. |
            """
        )

    st.markdown(
        """
        <div class="info-box">
            <strong>🔒 No Data Leakage:</strong> All preprocessing (imputation, scaling, encoding) is fitted
            only on the training set and applied to the test set. Cross-validation is performed on the
            training set only; the test set is used only for final held-out evaluation.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# Page: Dataset Explorer
# ===========================================================================
def page_dataset_explorer() -> None:
    st.markdown("## 📂 Dataset Explorer")

    # ----- Source selection -----
    source = st.radio(
        "Data source",
        ["📦 Built-in Sample Dataset", "📤 Upload CSV"],
        horizontal=True,
    )

    if source == "📦 Built-in Sample Dataset":
        choice = st.selectbox("Select a sample dataset", list(SAMPLE_DATASETS.keys()))
        if st.button("Load Dataset", type="primary"):
            with st.spinner("Loading…"):
                try:
                    loaded = DataLoader.load_sample_dataset(choice)
                    st.session_state.raw_df = loaded
                    st.session_state.raw_dataset_name = choice
                    st.session_state.df = loaded
                    st.session_state.dataset_name = choice
                    default_task = "regression" if "Regression" in choice else "classification"
                    st.session_state.target_col = loaded.columns[-1]
                    st.session_state.task_type = default_task
                    st.session_state.info = DataLoader.get_dataset_info(loaded, loaded.columns[-1], default_task)
                    # Reset downstream state
                    st.session_state.preprocessor = None
                    st.session_state.X = None
                    st.session_state.y = None
                    st.session_state.label_encoder = None
                    st.session_state.training_results = {}
                    st.session_state.selected_model = None
                    st.success(f"✅ Loaded **{choice}** – {loaded.shape[0]:,} rows × {loaded.shape[1]} cols")
                except Exception as exc:
                    st.error(f"Failed to load dataset: {exc}")
                    return
    else:
        uploaded = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            help="Maximum 50 MB. UTF-8 or Latin-1 encoding supported.",
        )
        if uploaded is not None:
            if uploaded.size > 50 * 1024 * 1024:
                st.error("File exceeds 50 MB limit.")
                return
            if st.session_state.raw_dataset_name != uploaded.name or st.session_state.df is None:
                with st.spinner("Parsing CSV…"):
                    try:
                        loaded = DataLoader.load_csv(uploaded)
                        st.session_state.raw_df = loaded
                        st.session_state.raw_dataset_name = uploaded.name
                        st.session_state.df = loaded
                        st.session_state.dataset_name = uploaded.name
                        st.session_state.target_col = loaded.columns[-1]
                        st.session_state.task_type = "classification"
                        st.session_state.info = DataLoader.get_dataset_info(loaded, loaded.columns[-1], "classification")
                        # Reset downstream state
                        st.session_state.preprocessor = None
                        st.session_state.X = None
                        st.session_state.y = None
                        st.session_state.label_encoder = None
                        st.session_state.training_results = {}
                        st.session_state.selected_model = None
                        st.success(f"✅ Uploaded **{uploaded.name}** – {loaded.shape[0]:,} rows × {loaded.shape[1]} cols")
                    except ValueError as exc:
                        st.error(str(exc))
                        return

    df: Optional[pd.DataFrame] = st.session_state.df
    dataset_name: Optional[str] = st.session_state.dataset_name

    if df is None:
        st.info("👆 Select a dataset and click 'Load Dataset' to get started.")
        return

    # ----- Target & task selection -----
    st.divider()
    st.markdown('<div class="section-header">Configure Task</div>', unsafe_allow_html=True)

    default_target_idx = (
        df.columns.get_loc(st.session_state.target_col)
        if st.session_state.target_col in df.columns
        else len(df.columns) - 1
    )
    default_task_idx = 0 if st.session_state.task_type == "classification" else 1

    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox(
            "Target column (what you want to predict)",
            options=df.columns.tolist(),
            index=default_target_idx,
        )
    with col2:
        task_type = st.selectbox(
            "Task type",
            options=["classification", "regression"],
            index=default_task_idx,
        )

    if target_col != st.session_state.target_col or task_type != st.session_state.task_type:
        st.session_state.target_col = target_col
        st.session_state.task_type = task_type
        # Clear downstream on task change
        st.session_state.preprocessor = None
        st.session_state.X = None
        st.session_state.y = None
        st.session_state.label_encoder = None
        st.session_state.training_results = {}
        st.session_state.selected_model = None

    # ----- Validation -----
    val = DataLoader.validate_dataset(df, target_col, task_type)
    if val.errors:
        for err in val.errors:
            st.error(f"❌ {err}")
        return
    if val.warnings:
        with st.expander("⚠️ Validation Warnings", expanded=True):
            for w in val.warnings:
                st.warning(w)

    # ----- Dataset Info -----
    info = DataLoader.get_dataset_info(df, target_col, task_type)
    st.session_state.info = info

    st.divider()
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card_html("Rows", f"{info.n_rows:,}", ""), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card_html("Columns", str(info.n_cols), "", color="#06B6D4"), unsafe_allow_html=True)
    with c3:
        st.markdown(
            kpi_card_html(
                "Missing Values",
                str(sum(info.missing_counts.values())),
                "",
                color="#F59E0B" if sum(info.missing_counts.values()) > 0 else "#10B981",
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            kpi_card_html(
                "Duplicate Rows",
                str(info.duplicate_rows),
                "",
                color="#EF4444" if info.duplicate_rows > 0 else "#10B981",
            ),
            unsafe_allow_html=True,
        )

    st.divider()

    tab_data, tab_types, tab_stats, tab_missing = st.tabs(
        ["📋 Sample Data", "🔤 Column Types", "📊 Statistics", "🕳️ Missing Values"]
    )

    with tab_data:
        st.dataframe(df.head(100), use_container_width=True)

    with tab_types:
        dtypes_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Dtype": df.dtypes.astype(str).values,
                "Unique Values": df.nunique().values,
                "Missing (%)": [f"{info.missing_pct[c]:.1f}%" for c in df.columns],
                "Type": [
                    "Numeric" if c in info.numeric_cols
                    else "Categorical/Text"
                    for c in df.columns
                ],
            }
        )
        st.dataframe(dtypes_df, use_container_width=True)

    with tab_stats:
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    with tab_missing:
        st.plotly_chart(plot_missing_values(df), use_container_width=True)

    # ----- Persist to session state -----
    st.divider()
    if st.button("✅ Confirm Dataset & Configuration", type="primary"):
        st.session_state.df = df
        st.session_state.dataset_name = dataset_name
        st.session_state.target_col = target_col
        st.session_state.task_type = task_type
        st.session_state.info = info
        # Reset downstream state
        st.session_state.preprocessor = None
        st.session_state.X = None
        st.session_state.y = None
        st.session_state.label_encoder = None
        st.session_state.training_results = {}
        st.session_state.selected_model = None
        st.success(
            f"Dataset confirmed! Target: **{target_col}** | Task: **{task_type}**. "
            "Proceed to Data Cleaning & EDA."
        )


# ===========================================================================
# Page: Data Cleaning & EDA
# ===========================================================================
def page_eda() -> None:
    st.markdown("## 🔬 Data Cleaning & EDA")

    if st.session_state.df is None:
        st.warning("⚠️ Please load a dataset first in **📂 Dataset Explorer**.")
        return

    df: pd.DataFrame = st.session_state.df
    target_col: str = st.session_state.target_col or df.columns[-1]
    task_type: str = st.session_state.task_type or "classification"
    if st.session_state.info is None:
        st.session_state.info = DataLoader.get_dataset_info(df, target_col, task_type)
    info: DatasetInfo = st.session_state.info

    # ----- Preprocessing Config -----
    st.markdown('<div class="section-header">Preprocessing Settings</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Settings here configure the sklearn Pipeline that will be used during model training. '
        'Preprocessing is fitted only on training data to prevent leakage.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        numeric_imputer = st.selectbox(
            "Numeric imputation strategy",
            ["mean", "median", "most_frequent"],
            help="Strategy for filling missing numeric values.",
        )
        scaler = st.selectbox(
            "Feature scaling",
            ["standard", "minmax", "none"],
            help="StandardScaler: zero mean, unit variance. MinMax: [0,1] range.",
        )
    with col2:
        categorical_imputer = st.selectbox(
            "Categorical imputation",
            ["most_frequent", "constant"],
        )
        encoder = st.selectbox(
            "Categorical encoding",
            ["onehot", "ordinal"],
            help="OneHot creates binary columns. Ordinal assigns integers.",
        )
    with col3:
        drop_all_nan = st.checkbox("Drop fully-missing columns", value=True)
        drop_const = st.checkbox("Drop constant columns", value=True)

    cfg = PreprocessingConfig(
        numeric_imputer=numeric_imputer,
        categorical_imputer=categorical_imputer,
        scaler=scaler,
        encoder=encoder,
        drop_all_nan_cols=drop_all_nan,
        drop_const_cols=drop_const,
    )

    if st.button("🔧 Apply Preprocessing Config", type="primary"):
        with st.spinner("Building preprocessing pipeline…"):
            try:
                preprocessor = Preprocessor(config=cfg)
                X, y_raw = preprocessor.prepare_features(df, target_col)
                y, le = preprocessor.encode_target(y_raw, task_type)
                preprocessor.build_pipeline(X)

                st.session_state.preprocessor = preprocessor
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.label_encoder = le
                # Reset models on config change
                st.session_state.training_results = {}
                st.session_state.selected_model = None

                report = preprocessor.report_
                st.success("✅ Preprocessing pipeline built successfully!")

                # Show what was done
                with st.expander("Preprocessing Report", expanded=True):
                    if report.dropped_columns:
                        st.warning(f"Dropped columns: {report.dropped_columns}")
                    st.info(
                        f"Numeric columns ({len(report.numeric_columns)}): "
                        f"{report.numeric_columns[:8]}{'…' if len(report.numeric_columns) > 8 else ''}"
                    )
                    if report.categorical_columns:
                        st.info(
                            f"Categorical columns ({len(report.categorical_columns)}): "
                            f"{report.categorical_columns}"
                        )
                    st.info(f"Scaling: {scaler} | Encoding: {encoder}")
                    if task_type == "classification" and le is not None:
                        st.info(f"Target classes: {le.classes_.tolist()}")

            except Exception as exc:
                st.error(f"Preprocessing failed: {exc}")
                logger.error(traceback.format_exc())
                return

    if st.session_state.preprocessor is None:
        try:
            preprocessor = Preprocessor(config=cfg)
            X, y_raw = preprocessor.prepare_features(df, target_col)
            y, le = preprocessor.encode_target(y_raw, task_type)
            preprocessor.build_pipeline(X)
            st.session_state.preprocessor = preprocessor
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.label_encoder = le
            X_display = X
        except Exception:
            X_display = df.drop(columns=[target_col], errors="ignore")
    else:
        X_display = st.session_state.X

    st.divider()

    # ----- EDA Tabs -----
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab_dist, tab_hist, tab_box, tab_corr, tab_feat, tab_outlier = st.tabs(
        ["Target Dist.", "Histograms", "Box Plots", "Correlation", "Feature vs Target", "Outliers"]
    )

    with tab_dist:
        st.plotly_chart(
            plot_target_distribution(df, target_col, task_type), use_container_width=True
        )
        if task_type == "classification":
            st.plotly_chart(
                plot_class_imbalance(df, target_col), use_container_width=True
            )

    with tab_hist:
        num_cols = X_display.select_dtypes(include=[np.number]).columns.tolist()
        st.plotly_chart(plot_histograms(X_display, num_cols), use_container_width=True)

    with tab_box:
        num_cols = X_display.select_dtypes(include=[np.number]).columns.tolist()
        st.plotly_chart(plot_boxplots(X_display, num_cols), use_container_width=True)

    with tab_corr:
        corr_df = df.select_dtypes(include=[np.number])
        st.plotly_chart(
            plot_correlation_heatmap(corr_df, target_col if target_col in corr_df.columns else None),
            use_container_width=True,
        )

    with tab_feat:
        num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
        feat_options = [c for c in df.columns if c != target_col]
        if feat_options:
            selected_feat = st.selectbox("Select a feature to compare with target", feat_options)
            if pd.api.types.is_numeric_dtype(df[selected_feat]):
                st.plotly_chart(
                    plot_feature_vs_target(df, selected_feat, target_col, task_type),
                    use_container_width=True,
                )
            else:
                st.info("Select a numeric feature for this chart.")

        # Pair plot for top numeric cols
        if len(num_cols_all) >= 2:
            with st.expander("Scatter Matrix (top 5 numeric features)"):
                top5 = num_cols_all[:5]
                st.plotly_chart(
                    plot_pairplot(df, top5, target_col if task_type == "classification" else None),
                    use_container_width=True,
                )

    with tab_outlier:
        num_cols = X_display.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            outliers = Preprocessor.detect_outliers_iqr(X_display, num_cols)
            outlier_df = pd.DataFrame(
                [(col, cnt, f"{cnt / len(X_display) * 100:.1f}%")
                 for col, cnt in outliers.items() if cnt > 0],
                columns=["Feature", "Outlier Count", "Outlier %"],
            )
            if outlier_df.empty:
                st.markdown(
                    '<div class="success-box">✅ No significant outliers detected (IQR method).</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.dataframe(outlier_df, use_container_width=True)
                st.caption(
                    "Outliers identified using the IQR method (1.5 × IQR beyond Q1/Q3). "
                    "Tree-based models are robust to outliers; linear models may be affected."
                )


# ===========================================================================
# Page: Model Training
# ===========================================================================
def page_model_training() -> None:
    st.markdown("## 🤖 Model Training")

    if st.session_state.df is None:
        st.warning("⚠️ Please load a dataset in **📂 Dataset Explorer** first.")
        return
    if st.session_state.preprocessor is None:
        st.warning("⚠️ Please apply preprocessing in **🔬 Data Cleaning & EDA** first.")
        return

    task_type: str = st.session_state.task_type
    model_registry = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS

    st.markdown('<div class="section-header">Training Configuration</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        selected_models = st.multiselect(
            "Select models to train",
            options=list(model_registry.keys()),
            default=list(model_registry.keys()),
            help="At least two models recommended for comparison.",
        )
    with col2:
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, step=0.05,
                              help="Fraction of data held out for final evaluation.")
        cv_folds = st.slider("Cross-validation folds", 3, 10, 5,
                             help="Number of folds for k-fold CV on the training set.")

    if len(selected_models) < 1:
        st.warning("Select at least one model.")
        return

    st.markdown(
        f"""
        <div class="info-box">
        <strong>Training plan:</strong> {len(selected_models)} model(s) will be trained.
        Random seed: 42 | Test size: {test_size:.0%} | CV folds: {cv_folds}<br>
        Cross-validation is performed on the <em>training set only</em>.
        The test set is used only for final held-out evaluation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚀 Train Models", type="primary"):
        X: pd.DataFrame = st.session_state.X
        y: np.ndarray = st.session_state.y

        if X is None or y is None:
            st.error("Feature matrix not found. Re-apply preprocessing.")
            return

        progress = st.progress(0)
        status = st.empty()
        results: dict[str, TrainingResult] = {}

        for i, model_name in enumerate(selected_models):
            status.text(f"Training {model_name} …")
            try:
                trainer = ModelTrainer(
                    preprocessor_pipeline=st.session_state.preprocessor.build_pipeline(X),
                    task_type=task_type,
                    selected_models=[model_name],
                    test_size=test_size,
                    cv_folds=cv_folds,
                    label_encoder=st.session_state.label_encoder,
                )
                model_results = trainer.train_all(X, y)
                results.update(model_results)
            except Exception as exc:
                st.error(f"Training failed for {model_name}: {exc}")
                logger.error(traceback.format_exc())
            progress.progress((i + 1) / len(selected_models))

        status.empty()
        progress.empty()

        st.session_state.training_results = results
        st.session_state.selected_model = list(results.keys())[0] if results else None

        if results:
            st.success(f"✅ Trained {len(results)} model(s) successfully!")

            # Quick results table
            comparison = compare_models(results, task_type)
            st.dataframe(comparison, use_container_width=True)

            for name, result in results.items():
                if result.warnings:
                    for w in result.warnings:
                        st.warning(f"**{name}**: {w}")

    # ----- Show existing results if already trained -----
    if st.session_state.training_results:
        st.divider()
        st.markdown('<div class="section-header">Cross-Validation Results</div>', unsafe_allow_html=True)
        st.caption("CV is performed on the training set only. Test results are shown in Model Comparison.")

        for name, result in st.session_state.training_results.items():
            with st.expander(f"📈 {name} – CV Summary", expanded=False):
                cv_summary = ModelEvaluator.summarise_cv(result.cv_results, task_type)
                if not cv_summary.empty:
                    st.dataframe(cv_summary, use_container_width=True)
                st.plotly_chart(
                    ModelEvaluator.plot_cv_scores(result.cv_results, name, task_type),
                    use_container_width=True,
                )


# ===========================================================================
# Page: Model Comparison
# ===========================================================================
def page_model_comparison() -> None:
    st.markdown("## 📊 Model Comparison")

    if not st.session_state.training_results:
        st.warning("⚠️ No trained models found. Go to **🤖 Model Training** first.")
        return

    results = st.session_state.training_results
    task_type = st.session_state.task_type

    comparison = compare_models(results, task_type)
    st.markdown('<div class="section-header">Held-Out Test Set Metrics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">These metrics are computed on the held-out test set that was never used '
        'during training or cross-validation.</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(comparison.style.highlight_max(axis=0, subset=comparison.columns[1:], color="#DBEAFE"),
                 use_container_width=True)

    st.plotly_chart(plot_model_comparison(comparison, task_type), use_container_width=True)

    st.divider()
    st.markdown('<div class="section-header">Per-Model Detailed Evaluation</div>', unsafe_allow_html=True)

    model_name = st.selectbox("Select model to inspect", list(results.keys()))
    result = results[model_name]

    if task_type == "classification":
        metrics = ModelEvaluator.compute_classification_metrics(result)

        # KPI row
        cols = st.columns(5)
        kpi_data = metrics.to_dict()
        colors = ["#4F46E5", "#06B6D4", "#10B981", "#F59E0B", "#8B5CF6"]
        for (label, val), col, color in zip(kpi_data.items(), cols, colors):
            with col:
                v = f"{val:.4f}" if isinstance(val, float) else str(val)
                st.markdown(kpi_card_html(label, v, "", color), unsafe_allow_html=True)

        st.divider()
        col_cm, col_roc = st.columns(2)
        with col_cm:
            st.plotly_chart(
                ModelEvaluator.plot_confusion_matrix(metrics, model_name),
                use_container_width=True,
            )
        with col_roc:
            roc_fig = ModelEvaluator.plot_roc_curve(result, model_name)
            if roc_fig:
                st.plotly_chart(roc_fig, use_container_width=True)
            else:
                st.info("ROC curve not available for this model/configuration.")

    else:
        metrics = ModelEvaluator.compute_regression_metrics(result)
        kpi_data = metrics.to_dict()
        cols = st.columns(3)
        colors = ["#4F46E5", "#EF4444", "#10B981"]
        for (label, val), col, color in zip(kpi_data.items(), cols, colors):
            with col:
                st.markdown(kpi_card_html(label, f"{val:.4f}", "", color), unsafe_allow_html=True)

        st.divider()
        col_resid, col_avp = st.columns(2)
        with col_resid:
            st.plotly_chart(
                ModelEvaluator.plot_residuals(result, model_name),
                use_container_width=True,
            )
        with col_avp:
            st.plotly_chart(
                ModelEvaluator.plot_actual_vs_predicted(result, model_name),
                use_container_width=True,
            )

    # Set selected model
    st.session_state.selected_model = model_name


# ===========================================================================
# Page: Prediction Playground
# ===========================================================================
def page_prediction_playground() -> None:
    st.markdown("## 🎯 Prediction Playground")

    if not st.session_state.training_results:
        st.warning("⚠️ No trained models found. Go to **🤖 Model Training** first.")
        return

    results = st.session_state.training_results
    task_type = st.session_state.task_type
    df_full: pd.DataFrame = st.session_state.df

    col_left, col_right = st.columns([1, 1])

    with col_left:
        model_name = st.selectbox("Select model", list(results.keys()))
        result = results[model_name]

        # Build prediction engine
        engine = PredictionEngine(
            pipeline=result.pipeline,
            feature_names=result.feature_names_in,
            task_type=task_type,
            label_encoder=st.session_state.label_encoder,
            training_data=df_full,
        )

        st.markdown("### Input Features")
        st.caption("Fill in the feature values below. Leave a field blank to let the model use imputation.")

        user_inputs: dict[str, Any] = {}
        feature_names = result.feature_names_in

        with st.form("prediction_form"):
            # Numeric features
            numeric_feats = [
                f for f in feature_names
                if f in df_full.columns and pd.api.types.is_numeric_dtype(df_full[f])
            ]
            cat_feats = [
                f for f in feature_names
                if f in df_full.columns and not pd.api.types.is_numeric_dtype(df_full[f])
            ]

            if numeric_feats:
                st.markdown("**Numeric Features**")
                n_per_row = 3
                for i in range(0, len(numeric_feats), n_per_row):
                    row_feats = numeric_feats[i:i + n_per_row]
                    cols = st.columns(len(row_feats))
                    for feat, col in zip(row_feats, cols):
                        with col:
                            median_val = float(df_full[feat].median()) if feat in df_full.columns else 0.0
                            user_inputs[feat] = st.number_input(
                                feat,
                                value=median_val,
                                format="%.4f",
                                key=f"pred_{feat}",
                            )

            if cat_feats:
                st.markdown("**Categorical Features**")
                for feat in cat_feats:
                    options = sorted(df_full[feat].dropna().unique().tolist())
                    user_inputs[feat] = st.selectbox(feat, options, key=f"pred_{feat}")

            submitted = st.form_submit_button("🔮 Predict", type="primary")

        if submitted:
            with st.spinner("Running prediction…"):
                output = engine.predict(user_inputs)

            if output.errors:
                for e in output.errors:
                    st.error(e)
            else:
                with col_right:
                    st.markdown("### Prediction Result")

                    if task_type == "classification":
                        st.markdown(
                            f"""
                            <div style="background:linear-gradient(135deg,#4F46E518,#4F46E505);
                                border:2px solid #4F46E540;border-radius:16px;padding:28px;text-align:center;">
                                <div style="font-size:0.8rem;color:#64748B;font-weight:600;text-transform:uppercase;">Predicted Class</div>
                                <div style="font-size:2.8rem;font-weight:800;color:#4F46E5;margin:8px 0;">{output.predicted_label}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if output.probabilities:
                            st.plotly_chart(
                                plot_probability_bar(output.probabilities),
                                use_container_width=True,
                            )
                    else:
                        st.markdown(
                            f"""
                            <div style="background:linear-gradient(135deg,#10B98118,#10B98105);
                                border:2px solid #10B98140;border-radius:16px;padding:28px;text-align:center;">
                                <div style="font-size:0.8rem;color:#64748B;font-weight:600;text-transform:uppercase;">Predicted Value</div>
                                <div style="font-size:2.8rem;font-weight:800;color:#10B981;margin:8px 0;">{float(output.predicted_value):.4f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    if output.warnings:
                        st.divider()
                        st.markdown("**⚠️ Prediction Warnings**")
                        for w in output.warnings:
                            st.warning(w)

                    st.markdown(
                        """
                        <div class="info-box" style="margin-top:16px;">
                        <strong>ℹ️ Model Limitations:</strong> Predictions are based on patterns in training data.
                        The model may not generalise to distributions significantly different from training data.
                        Always validate predictions with domain expertise before acting on them.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ===========================================================================
# Page: Model Insights
# ===========================================================================
def page_model_insights() -> None:
    st.markdown("## 🧠 Model Insights")

    if not st.session_state.training_results:
        st.warning("⚠️ No trained models found. Go to **🤖 Model Training** first.")
        return

    results = st.session_state.training_results
    model_name = st.selectbox("Select model", list(results.keys()))
    result = results[model_name]

    # Feature names after transformation
    preprocessor: Preprocessor = st.session_state.preprocessor
    X = st.session_state.X
    feature_names_transformed: list[str]
    try:
        feature_names_transformed = preprocessor.get_feature_names(X)
    except Exception:
        feature_names_transformed = result.feature_names_in

    st.divider()

    # ----- Built-in importance -----
    builtin_imp = ModelEvaluator.get_feature_importance(result, feature_names_transformed)

    tab_builtin, tab_perm = st.tabs(["🌲 Built-in Feature Importance", "🔀 Permutation Importance"])

    with tab_builtin:
        if builtin_imp is not None:
            st.markdown(
                '<div class="info-box">'
                '<strong>Built-in Importance</strong> (tree models only): measures how much each feature '
                'reduces impurity across all splits. Available for Decision Tree and Random Forest. '
                '<em>Correlation ≠ causation.</em></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                ModelEvaluator.plot_feature_importance(
                    builtin_imp, f"Feature Importance – {model_name}"
                ),
                use_container_width=True,
            )
            st.dataframe(builtin_imp.head(20), use_container_width=True)
        else:
            st.info(
                f"**{model_name}** does not expose built-in feature importances. "
                "Use the Permutation Importance tab instead."
            )

    with tab_perm:
        st.markdown(
            '<div class="info-box">'
            '<strong>Permutation Importance</strong>: model-agnostic. Measures the average decrease '
            'in test-set score when each feature is randomly shuffled. '
            'Negative values indicate the feature adds noise. <em>Correlation ≠ causation.</em>'
            '</div>',
            unsafe_allow_html=True,
        )

        n_repeats = st.slider("Number of permutation repeats", 5, 30, 10)
        if st.button("Compute Permutation Importance", key="perm_imp_btn"):
            with st.spinner("Computing permutation importance (this may take a moment)…"):
                perm_imp = ModelEvaluator.get_permutation_importance(
                    result, result.feature_names_in, n_repeats=n_repeats
                )
            if perm_imp is not None:
                st.plotly_chart(
                    ModelEvaluator.plot_feature_importance(
                        perm_imp, f"Permutation Importance – {model_name}", top_n=20
                    ),
                    use_container_width=True,
                )
                st.dataframe(perm_imp.head(20), use_container_width=True)
            else:
                st.error("Permutation importance computation failed. Check logs.")

    # ----- Limitations -----
    st.divider()
    st.markdown('<div class="section-header">Model Limitations & Warnings</div>', unsafe_allow_html=True)

    estimator = result.pipeline.named_steps.get("estimator")
    estimator_name = type(estimator).__name__ if estimator else "Unknown"

    limitation_map = {
        "LogisticRegression": [
            "Assumes a linear decision boundary.",
            "May underperform on non-linear data without feature engineering.",
            "Sensitive to feature scaling (StandardScaler applied).",
        ],
        "DecisionTreeClassifier": [
            "Prone to overfitting when depth is not limited.",
            "Unstable – small data changes can produce a very different tree.",
            "Not great for extrapolation.",
        ],
        "RandomForestClassifier": [
            "Harder to interpret than a single decision tree.",
            "Slower to train and predict than simpler models.",
            "May overfit on noisy datasets with very many features.",
        ],
        "LinearRegression": [
            "Assumes a linear relationship between features and target.",
            "Sensitive to outliers.",
            "Assumes homoscedasticity and no multicollinearity.",
        ],
        "DecisionTreeRegressor": [
            "Cannot extrapolate outside the training target range.",
            "Prone to overfitting.",
        ],
        "RandomForestRegressor": [
            "Cannot extrapolate outside the training target range.",
            "Computationally intensive for very large datasets.",
        ],
    }

    limitations = limitation_map.get(estimator_name, ["No specific limitation notes available."])
    for lim in limitations:
        st.markdown(
            f'<div class="warn-box">⚠️ {lim}</div>',
            unsafe_allow_html=True,
        )

    if result.warnings:
        for w in result.warnings:
            st.markdown(f'<div class="warn-box">⚠️ {w}</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box" style="margin-top:12px;">
        <strong>General Reminder:</strong> Feature importance scores reflect the model's internal associations
        with training data patterns. They <em>do not</em> imply causation, and should not be used to draw
        causal conclusions without further domain analysis and controlled experimentation.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# Router
# ===========================================================================
PAGE_ROUTING = {
    "Overview": page_overview,
    "Dataset Explorer": page_dataset_explorer,
    "Data Cleaning & EDA": page_eda,
    "Model Training": page_model_training,
    "Model Comparison": page_model_comparison,
    "Prediction Playground": page_prediction_playground,
    "Model Insights": page_model_insights,
}

PAGE_ROUTING[page_key]()
