# 📊 Customer Churn Prediction & Retention Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB1222?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> **An end-to-end Machine Learning and Business Intelligence solution** designed to proactively identify at-risk retail banking customers, quantify individual attrition probabilities, diagnose key behavioral churn drivers, and empower relationship managers through an interactive Streamlit retention dashboard.

---

## 📑 Table of Contents

- [🌟 Executive Summary](#-executive-summary)
- [🎯 Business Problem & Core Objectives](#-business-problem--core-objectives)
- [🏗️ System Architecture & Workflow](#️-system-architecture--workflow)
- [🔍 Exploratory Data Analysis & Critical Insights](#-exploratory-data-analysis--critical-insights)
- [🔬 Data Preprocessing & Leakage-Safe Pipeline](#-data-preprocessing--leakage-safe-pipeline)
- [🤖 Model Benchmarking & Experimental Results](#-model-benchmarking--experimental-results)
- [🖥️ Streamlit Web Application Walkthrough](#️-streamlit-web-application-walkthrough)
- [📁 Project Structure](#-project-structure)
- [⚡ Quickstart & Installation](#-quickstart--installation)
- [🛠️ How to Run & Reproduce](#️-how-to-run--reproduce)
- [📈 Strategic Business Retention Playbook](#-strategic-business-retention-playbook)
- [👥 Author & License](#-author--license)

---

## 🌟 Executive Summary

In retail banking, acquiring a new customer costs **5 to 7 times more** than retaining an existing one. Unplanned customer attrition directly erodes total deposits, reduces interchange fees, and damages long-term Customer Lifetime Value (CLV).

This repository contains a full-lifecycle Machine Learning and Decision Support System trained and evaluated on **10,000 customer banking profiles**. It benchmarks **6 classification algorithms** across two rigorous validation regimes (with and without customer service complaints), pairs them with a **leakage-safe preprocessing pipeline**, and packages the final solution into a production-ready **Streamlit web application**.

```
  ┌───────────────────────────┐      ┌───────────────────────────┐
  │   10,000 Customer Data    │ ───► │  Exploratory Data Science │
  │ (Demographic & Financial) │      │  (7 In-depth EDA Notebooks)│
  └───────────────────────────┘      └───────────────────────────┘
                │
                ▼
  ┌───────────────────────────┐      ┌───────────────────────────┐
  │  Leakage-Safe Preprocess  │ ───► │   6-Model ML Tournament   │
  │ (IQR Clip + OHE + Scaler) │      │  (LR, DT, RF, GB, KNN, XGB)│
  └───────────────────────────┘      └───────────────────────────┘
                │
                ▼
  ┌──────────────────────────────────────────────────────────────┐
  │             Interactive Streamlit Web Dashboard              │
  │   • Batch CSV Scoring Engine  • Real-Time KPI Metric Cards   │
  │   • 3-Tier Risk Segmentation  • 1-Click Scored CSV Export    │
  └──────────────────────────────────────────────────────────────┘
```

---

## 🎯 Business Problem & Core Objectives

### The Challenge
A leading international bank is experiencing customer attrition across its European branches (France, Germany, Spain). Without predictive modeling, relationship teams only discover a customer has left *after* the account is closed—when it is already too late to intervene.

### Core Objectives
1. **Accurate Attrition Probability**: Predict the precise likelihood ($0.0\% - 100.0\%$) of each customer churning before they disengage.
2. **Actionable Risk Segmentation**:
   - 🔴 **High Risk ($\ge 70\%$ probability)**: Immediate priority outreach by relationship managers with customized retention packages.
   - 🟡 **Medium Risk ($40\% - 69\%$ probability)**: Automated engagement, targeted loyalty incentives, and fee waivers.
   - 🟢 **Low Risk ($< 40\%$ probability)**: Standard marketing communications and relationship nurturing.
3. **Root Cause Diagnosis**: Surface key demographic and behavioral indicators (e.g., complaint filings, balance thresholds, age cohorts, product portfolio size).
4. **Self-Service Decision Tooling**: Provide branch staff and retention analysts with an intuitive web dashboard capable of scoring batch customer files in seconds without writing code.

---

## 🏗️ System Architecture & Workflow

The system is built on modular Python packages, separating data loading, statistical profiling, feature engineering, model training, evaluation, hyperparameter tuning, and web serving.


<img width="3438" height="3086" alt="diagram" src="https://github.com/user-attachments/assets/283c6160-e33c-4559-8c87-b09467935acf" />


---

## 🔍 Exploratory Data Analysis & Critical Insights

The dataset comprises **10,000 banking customers** with **18 attributes** spanning demographics, financial positions, account activity, and customer service touchpoints.

### Feature Taxonomy

| Category | Features | Description & Business Meaning |
| :--- | :--- | :--- |
| **Identifiers** | `RowNumber`, `CustomerId`, `Surname` | Unique account tracking identifiers (dropped from ML feature matrix) |
| **Demographic** | `Geography`, `Gender`, `Age` | Customer regional market (France, Germany, Spain), biological gender, customer age |
| **Financial Health** | `CreditScore`, `Balance`, `EstimatedSalary`, `Credit Card`, `Card Type` | Credit rating (350–850), ledger balance ($€$), annual estimated income, card ownership, card tier (DIAMOND, GOLD, PLATINUM, SILVER) |
| **Engagement** | `Tenure`, `NumOfProducts`, `IsActiveMember`, `Point Earned`, `Satisfaction Score` | Account tenure (years), number of bank products used (1–4), digital/branch activity flag, reward points accumulated, satisfaction rating (1–5) |
| **Service Touchpoint**| `Complain` | Customer logged a formal grievance / complaint ($0 = \text{No}, 1 = \text{Yes}$) |
| **Target Variable** | `Churned` | Ground truth churn flag ($0 = \text{Retained } [79.62\%], 1 = \text{Churned } [20.38\%]$) |

---

### 💡 5 Golden Insights from In-Depth EDA

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                            MAJOR DATA & BUSINESS FINDINGS                            ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║ 1. 🚨 The Complaint Catalyst (Strongest Churn Driver):                               ║
║    • Customers who lodged a complaint (Complain = 1) had a 99.51% churn rate.        ║
║    • Customers with zero complaints (Complain = 0) had a 0.05% churn rate.           ║
║    • Correlation with Churn: r = +0.9957 (Near-deterministic indicator).             ║
║                                                                                      ║
║ 2. 🇩🇪 The German Market Anomaly:                                                     ║
║    • German accounts churn at 39.9%—more than double France (16.1%) and Spain (16.7%)║
║    • German customers maintain significantly higher average account balances.        ║
║                                                                                      ║
║ 3. 👥 Age Cohort Vulnerability:                                                      ║
║    • Customers aged 45–60 churn at ~56.2% (peak attrition zone).                     ║
║    • Younger demographics (18–35) exhibit high retention (churn rate < 9.5%).        ║
║                                                                                      ║
║ 4. 📦 The Multi-Product Paradox:                                                     ║
║    • Customers with 1 product: ~27.7% churn rate.                                    ║
║    • Customers with 2 products: 7.6% churn rate (The optimal sweet-spot!).           ║
║    • Customers with 3 or 4 products: >82% churn rate (severe product friction).     ║
║                                                                                      ║
║ 5. 🛡️ The Active Engagement Shield:                                                  ║
║    • Active members (IsActiveMember = 1) are ~50% less likely to leave than          ║
║      inactive members (14.3% vs 26.8% churn rate).                                   ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 🔬 Data Preprocessing & Leakage-Safe Pipeline

Data leakage can artificially inflate evaluation scores while causing models to fail in production. To ensure absolute data integrity, all transformations learn their statistical parameters **strictly on the training partition** and apply the learned rules to test/inference data.

1. **Header Normalization & Cleaning** (`column_stand`):
   - Trims whitespace, converts to lowercase, and standardizes spacing/dashes (`Credit Score` $\rightarrow$ `creditscore`).
2. **Missing Value Imputation** (`fill_num`, `fill_cat`):
   - **Continuous Features**: Imputed using **Median** strategy via `SimpleImputer`.
   - **Categorical Features**: Imputed using **Most Frequent (Mode)** strategy.
3. **IQR-Based Outlier Clipping** (`remove_outlier` & `IQRClipper`):
   - Computes first quartile ($Q_1$) and third quartile ($Q_3$).
   - Determines clipping boundaries: $[\text{lower} = Q_1 - 1.5 \times IQR, \; \text{upper} = Q_3 + 1.5 \times IQR]$.
   - Caps extreme outliers without dropping rows or distorting distributions.
4. **Categorical Encoding** (`one_hot`):
   - Encodes nominal attributes (`geography`, `gender`, `card_type`) using `OneHotEncoder(handle_unknown="ignore")`.
5. **Feature Scaling** (`stand_scaler`):
   - Standardizes continuous numeric features (`balance`, `estimatedsalary`, `creditscore`, `age`, `point_earned`, `tenure`) to zero mean and unit variance using `StandardScaler`.
6. **Production Schema Validation & Alias Engine** (`app.py`):
   - Automatically maps common CSV column variants (e.g., `HasCrCard` $\rightarrow$ `credit_card`, `NumOfProducts` $\rightarrow$ `numofproducts`).
   - Verifies all required numeric and categorical features before passing rows to the prediction engine.

---

## 🤖 Model Benchmarking & Experimental Results

To examine predictive behavior with full transparency, models were benchmarked across two distinct experimental environments:
1. **With `Complain` (Production Serving Setup)**: When customer service ticket records are available at prediction time.
2. **Without `Complain` (Early-Warning Behavioral Setup)**: Testing pure demographic and financial signals before a complaint is escalated.

### 📊 5-Fold Stratified Cross-Validation Benchmark (8,000 Training Records)

| Model Architecture | Feature Regime | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | *Without Complain* | **0.8608** | 0.7617 | **0.4601** | **0.5735** | **0.8622** |
| **Random Forest** | *Without Complain* | 0.8603 | **0.7865** | 0.4319 | 0.5571 | 0.8473 |
| **XGBoost Classifier** | *Without Complain* | 0.8481 | 0.6787 | 0.4859 | 0.5658 | 0.8351 |
| **Logistic Regression** | *Without Complain* | 0.8172 | 0.6311 | 0.2497 | 0.3575 | 0.7712 |
| **K-Nearest Neighbors** | *Without Complain* | 0.8066 | 0.5552 | 0.2497 | 0.3442 | 0.7085 |
| **Decision Tree** | *Without Complain* | 0.7831 | 0.4696 | 0.4926 | 0.4806 | 0.6750 |
| | | | | | | |
| **Logistic Regression** 🏆 | *With Complain* | **0.9986** | **0.9945** | **0.9988** | **0.9966** | **0.9994** |
| **Random Forest** | *With Complain* | 0.9986 | 0.9945 | 0.9988 | 0.9966 | 0.9992 |
| **XGBoost Classifier** | *With Complain* | 0.9986 | 0.9945 | 0.9988 | 0.9966 | 0.9991 |
| **Gradient Boosting** | *With Complain* | 0.9984 | 0.9933 | 0.9988 | 0.9960 | 0.9987 |
| **Decision Tree** | *With Complain* | 0.9972 | 0.9933 | 0.9933 | 0.9933 | 0.9958 |
| **K-Nearest Neighbors** | *With Complain* | 0.9060 | 0.9480 | 0.5699 | 0.7113 | 0.9377 |

### 🎯 Held-Out Test Set Performance (2,000 Unseen Customers)

```
=============================================================================
  MODEL EVALUATION SUMMARY ON HELD-OUT TEST DATA (2,000 ROWS)
=============================================================================
  Model                Accuracy   Precision   Recall   F1-Score   ROC-AUC
  ─────────────────────────────────────────────────────────────────────────
  Logistic Regression   0.9985     0.9975     0.9951    0.9963    0.9992
  Random Forest         0.9985     0.9975     0.9951    0.9963    0.9976
  XGBoost               0.9985     0.9975     0.9951    0.9963    0.9970
  Gradient Boosting     0.9980     0.9951     0.9951    0.9951    0.9981
  Decision Tree         0.9965     0.9951     0.9877    0.9914    0.9932
  KNN                   0.9110     0.9563     0.5907    0.7303    0.9445
=============================================================================
```

> [!NOTE]
> **Logistic Regression** is persisted in `models/best_model.pkl` for production serving due to its optimal ROC-AUC (0.9994), instantaneous inference latency, and well-calibrated output probabilities.

---

## 🖥️ Streamlit Web Application Walkthrough

The web application (`app.py`) provides an interactive interface built for relationship managers, retention executives, and data scientists.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  📈 Customer Churn Prediction System                                             │
├──────────────────────────────────────────────────────────────────────────────────┤
│  [ 🏠 Dashboard ]    [ 📤 Predict Churn ]    [ 👥 Customer Results ]    [ ℹ️ About ] │
├──────────────────────────────────────────────────────────────────────────────────┤
│  Total: 10,000  │ Likely Churn: 2,038 │ Retained: 7,962 │ High Risk: 1,980 (19.8%)│
├──────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────┐  ┌───────────────────────────────────┐  │
│  │   Churn Prediction Distribution     │  │      Risk-Level Distribution      │  │
│  │   [██████████████████░░░░] 79.6%    │  │   [████░░░░░░░░░░░░░░░░] 19.8% High  │  │
│  │   [████░░░░░░░░░░░░░░░░░░] 20.4%    │  │   [████████████████░░░░] 79.9% Low   │  │
│  └─────────────────────────────────────┘  └───────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────────────────────┤
│  🔍 Filter: [ High Risk ▼ ]   Search: [ Mitchell | 15647311 | Germany          ] │
│  📥 [ Download Scored Customer Results (CSV) ]                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Key Modules & Capabilities

1. **🏠 Dashboard**:
   - **Executive KPI Cards**: Real-time summary metrics for Total Scored Accounts, Likely to Churn, Retained, High Risk Count, Average Churn Probability, and Predicted Attrition Rate.
   - **Visual Risk Histograms**: Probability bucket distributions (`0–20%`, `20–40%`, `40–70%`, `70–100%`) and classification share charts.

2. **📤 Predict Churn Workspace**:
   - **Flexible Ingestion**: Upload custom customer CSVs or load the built-in 700-row demonstration dataset with 1 click.
   - **Automated Schema Validator**: Verifies columns, maps aliases, displays missing feature warnings, and ensures robust type conversions.
   - **Instant Inference Engine**: Scores all customer rows using persisted pipeline artifacts.

3. **👥 Customer Results & Retention Console**:
   - **Risk-Ranked Account Table**: Orders customers by descending churn risk score.
   - **Multi-Filter Console**: Instant segment filtering (`High Risk`, `Medium Risk`, `Low Risk`, `Likely to Churn`, `All`).
   - **Omni-Search Engine**: Live search across Customer IDs, Surnames, and Geographical regions.
   - **Individual Customer Drawer**: Expandable diagnostic card showing customer metrics, identifiers, and complete feature values.
   - **1-Click CSV Export**: Download the full scored dataset with predicted labels, risk tiers, and probabilities for CRM integration.

4. **ℹ️ About & Documentation**:
   - Feature definitions, schema validation requirements, and risk threshold references.

---

## 📁 Project Structure

```bash
customer_churn_analysis/
├── app.py                      # Production Streamlit Web Dashboard
├── experiment_validation.py    # Leakage-safe 5-fold cross-validation experiment runner
├── requirements.txt            # Project dependencies and version pins
├── README.md                   # Comprehensive project documentation
├── agy.init                    # Antigravity Workspace Initialization Profile
│
├── data/
│   └── raw/
│       └── customer churn.csv  # 10,000-row raw retail banking dataset
│
├── models/                     # Serialized preprocessors and production model artifacts
│   ├── Standard_scaler.pkl     # Fitted StandardScaler (continuous numeric features)
│   ├── best_model.pkl          # Trained Logistic Regression classifier
│   ├── cat_imputer.pkl         # Fitted SimpleImputer (most_frequent categorical)
│   ├── encoder.pkl             # Fitted OneHotEncoder (geography, gender, card_type)
│   ├── num_imputer.pkl         # Fitted SimpleImputer (median numeric)
│   └── outlier_bonds.pkl       # Learned IQR clipping bounds dictionary
│
├── notebooks/                  # 7 Exploratory & statistical Jupyter notebooks
│   ├── 01_data_cleaning.ipynb  # Initial ingestion, missing value checks & summary stats
│   ├── univariate_analysis/
│   │   ├── categorical_analysis.ipynb # Distributions of categorical features
│   │   └── numerical_analysis.ipynb   # Distributions, skewness & outlier scans
│   ├── bivariate_analysis/
│   │   ├── categorical.ipynb          # Categorical features vs Churned target
│   │   ├── numerical.ipynb            # Numeric features vs numeric relationships
│   │   └── numerical&categorical.ipynb # Numeric distributions split across churn classes
│   └── multivariate_analysis/
│       └── multivariate_analysis.ipynb # Correlation heatmaps & one-hot feature analysis
│
└── src/                        # Modular source codebase
    ├── __init__.py             # Package initializer
    ├── constants.py            # Feature groups, target constants & hyperparameter grids
    ├── data_loader.py          # Data ingestion utility
    ├── statistics.py           # Comprehensive statistical profiling and summary generator
    ├── preprocessing.py        # Imputers, IQR clippers, scalers & categorical encoders
    ├── train.py                # Multi-model training executor loop
    ├── evaluate.py             # Classification metric evaluator (AUC, F1, Recall, etc.)
    ├── tuning.py               # GridSearchCV and RandomizedSearchCV tuning routines
    └── main.py                 # End-to-end training pipeline & artifact exporter
```

---

## ⚡ Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/anchitkaushal/customer-churn-prediction-system.git
cd customer_churn_analysis
```

### 2. Create and Activate a Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate on Linux/macOS:
source .venv/bin/activate

# Activate on Windows (cmd):
# .venv\Scripts\activate.bat

# Activate on Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🛠️ How to Run & Reproduce

### 🚀 1. Run the Streamlit Web Application
Launch the interactive decision dashboard in your browser:
```bash
streamlit run app.py
```
> The dashboard will automatically launch at `http://localhost:8501`.

### 🔄 2. Train the End-to-End Pipeline
Execute data loading, preprocessing, model training, evaluation, and serialize fresh `.pkl` artifacts into `models/`:
```bash
python -m src.main
```

### 🔬 3. Run Leakage-Safe Validation Experiments
Execute the 5-fold cross-validation experiment comparing model performance with and without `Complain`:
```bash
python experiment_validation.py
```

### 📓 4. Launch Jupyter Notebooks for Exploration
Inspect exploratory data analysis notebooks and statistical plots:
```bash
jupyter lab
```

---

## 📈 Strategic Business Retention Playbook

Based on empirical data findings and model explanations, bank leadership should implement the following targeted initiatives:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      4-PILLAR CUSTOMER RETENTION PLAYBOOK                        │
├──────────────────────────────────────────────────────────────────────────────────┤
│ 1. ⚡ The "Golden 2-Hour" Complaint Resolution Protocol                           │
│    • Insight: Customer complaint is the single highest predictor of churn (>99%). │
│    • Action: Any customer filing a grievance is instantly tagged as Priority-1.  │
│      Relationship managers must initiate direct resolution within 2 hours.       │
├──────────────────────────────────────────────────────────────────────────────────┤
│ 2. 🇩🇪 Targeted German Market Value Proposition                                   │
│    • Insight: German customers churn at 39.9% (2x other regions) with high funds. │
│    • Action: Introduce localized premium savings tiers, higher interest yield    │
│      products, and fee waivers for maintaining significant deposit balances.     │
├──────────────────────────────────────────────────────────────────────────────────┤
│ 3. 📦 Multi-Product Usability Redesign                                            │
│    • Insight: Accounts holding 3 or 4 products have an 80%+ attrition rate.      │
│    • Action: Audit cross-product UX. Simplify account management into a single   │
│      unified mobile interface to remove administrative complexity and friction. │
├──────────────────────────────────────────────────────────────────────────────────┤
│ 4. 🎯 Mid-Life Wealth Advisory Engagement (Ages 45–60)                           │
│    • Insight: 45–60 year old customers experience peak attrition (~56%).          │
│    • Action: Proactively assign dedicated financial advisors offering retirement │
│      planning, mortgage refinancing, and wealth management consultations.        │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 👥 Author & License

- **Author**: Anchit Kaushal
- **Project**: Customer Churn Analysis & Machine Learning Prediction System
- **License**: Distributed under the [MIT License](LICENSE).

⭐ *If you find this project valuable, please consider giving it a star on GitHub!*
