# ModelScope ML

A **production-quality Machine Learning Prediction and Model Evaluation Platform** built with Python, Streamlit, and Scikit-learn. Train, evaluate, compare, and deploy classification and regression models directly in your browser — no API keys or internet connection required.

---

## Features

- 📂 **Dataset Handling** – Built-in sample datasets (Iris, Breast Cancer, Wine, Diabetes) or upload your own CSV
- 🔬 **Data Cleaning & EDA** – Histograms, box plots, correlation heatmaps, target distributions, outlier detection
- 🤖 **Model Training** – Logistic Regression, Decision Tree, Random Forest (classification & regression variants)
- 📊 **Model Comparison** – Side-by-side metric table and interactive Plotly charts
- 🎯 **Prediction Playground** – Input features via a form; get predictions with probabilities
- 🧠 **Model Insights** – Feature importance (built-in + permutation), model limitations, and caveats
- 🔒 **No Data Leakage** – sklearn Pipelines + ColumnTransformer fitted only on training data

---

## Architecture

```
ModelScope ML/
├── app.py                    # Main Streamlit application (routing, UI, state)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # CSV loading, sample datasets, validation
│   ├── preprocessing.py      # sklearn Pipelines, imputation, scaling, encoding
│   ├── eda.py                # Plotly EDA charts
│   ├── model_trainer.py      # Model training with CV and train/test split
│   ├── evaluation.py         # Metrics, confusion matrix, ROC, feature importance
│   ├── prediction.py         # Prediction engine with input validation
│   └── visualizations.py     # Shared chart helpers and KPI cards
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_model_training.py
│   └── test_prediction.py
├── .streamlit/
│   └── config.toml           # Streamlit theme config
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation & Running

### Prerequisites
- Python 3.11+
- pip

### 1. Clone / download the project
```bash
git clone <repo-url>
cd "ModelScope ML"
```

### 2. (Optional) Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Dataset Instructions

### Built-in Datasets (no setup needed)
| Dataset | Task | Rows | Features |
|---|---|---|---|
| Iris | Classification (3 classes) | 150 | 4 numeric |
| Breast Cancer Wisconsin | Classification (binary) | 569 | 30 numeric |
| Wine Quality | Classification (3 classes) | 178 | 13 numeric |
| Diabetes | Regression | 442 | 10 numeric |

### Uploading a Custom CSV
- Format: UTF-8 or Latin-1 encoded CSV
- Maximum size: 50 MB
- Minimum: 10 rows × 2 columns
- One column must be the target (what you want to predict)
- Mixed numeric and categorical columns are supported

---

## Supported Algorithms

### Classification
| Model | Notes |
|---|---|
| Logistic Regression | Linear, fast, interpretable. Requires scaling. |
| Decision Tree | Non-linear, easy to visualise. Prone to overfitting. |
| Random Forest | Ensemble of trees. Robust, handles non-linearity. |

### Regression
| Model | Notes |
|---|---|
| Linear Regression | Assumes linear relationship. Interpretable. |
| Decision Tree Regressor | Non-linear. Cannot extrapolate beyond training range. |
| Random Forest Regressor | Ensemble. Robust. Cannot extrapolate beyond training range. |

---

## Evaluation Methodology

### Classification Metrics
| Metric | Description |
|---|---|
| Accuracy | Fraction of correct predictions. Simple but can be misleading on imbalanced data. |
| Precision (weighted) | Weighted average precision across all classes. |
| Recall (weighted) | Weighted average recall across all classes. |
| F1-Score (weighted) | Harmonic mean of precision and recall. Best for imbalanced classes. |
| ROC-AUC | Binary: standard AUC. Multi-class: One-vs-Rest weighted average. |
| Confusion Matrix | Visual breakdown of true/false positives and negatives per class. |

### Regression Metrics
| Metric | Description |
|---|---|
| MAE | Mean Absolute Error. Robust to outliers. |
| RMSE | Root Mean Squared Error. Penalises large errors. |
| R² | Coefficient of determination. 1.0 = perfect, 0 = predicts the mean, negative = worse than baseline. |

### Cross-Validation
- `CV_FOLDS = 5` (configurable in UI)
- `StratifiedKFold` for classification, `KFold` for regression
- Performed **only on the training set** — test set is never seen during CV or hyperparameter selection
- Test set metrics are always final held-out results

---

## Testing Instructions

```bash
# Run all tests
pytest tests/ -v

# Run a specific test module
pytest tests/test_data_loader.py -v

# Run with coverage (if coverage is installed)
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Capturing Screenshots

Once the app is running:
1. Open `http://localhost:8501`
2. Load the **Iris** dataset, click **Confirm**
3. Go to EDA → apply preprocessing
4. Go to Model Training → train all models
5. Navigate to each page and use browser screenshot tools (`Ctrl+Shift+S` or OS snipping tool)

---

## Deployment to Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository and set **Main file path** to `app.py`
5. Click **Deploy** — no secrets or environment variables required

---

## Known Limitations

- **No hyperparameter tuning UI** – models use sensible defaults (max_depth=6, n_estimators=100). Future enhancement: add GridSearchCV/RandomizedSearchCV.
- **Random Forest permutation importance** is slow on large datasets (>10k rows). The UI shows a progress spinner.
- **Regression models cannot extrapolate** beyond the range of training targets (Decision Tree, Random Forest). Linear Regression can extrapolate but may be inaccurate.
- **OHE increases dimensionality** for high-cardinality categorical features. Consider ordinal encoding or dropping such columns manually.
- **No time-series support** – train/test split is random; temporal datasets should use TimeSeriesSplit manually.
- **Model persistence** (save/load via Joblib) is ready in the model trainer but not yet exposed in the UI.
- **Multi-label classification** is not supported.

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Streamlit | ≥1.32 | Web interface |
| Pandas | ≥2.1 | Data manipulation |
| NumPy | ≥1.26 | Numerical computing |
| Scikit-learn | ≥1.4 | ML pipelines, models, metrics |
| Plotly | ≥5.18 | Interactive visualisations |
| Joblib | ≥1.3 | Model serialisation |
| pytest | ≥7.4 | Test suite |

---

*Built as a portfolio project demonstrating Python OOP, Pandas, Scikit-learn, feature engineering, model evaluation, and Streamlit UI development.*
