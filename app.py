"""Streamlit interface for batch customer churn predictions."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pandas.errors import EmptyDataError

from src.preprocessing import column_stand


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "customer churn.csv"
TARGET_COLUMN = "churned"
IDENTIFIER_COLUMNS = {"rownumber", "customerid", "surname"}
ARTIFACT_PATHS = {
    "model": MODEL_DIR / "best_model.pkl",
    "num_imputer": MODEL_DIR / "num_imputer.pkl",
    "cat_imputer": MODEL_DIR / "cat_imputer.pkl",
    "encoder": MODEL_DIR / "encoder.pkl",
    "scaler": MODEL_DIR / "Standard_scaler.pkl",
    "outlier_bounds": MODEL_DIR / "outlier_bonds.pkl",
}
COLUMN_ALIASES = {
    "credit_score": "creditscore",
    "number_of_products": "numofproducts",
    "num_of_products": "numofproducts",
    "numberofproducts": "numofproducts",
    "creditcard": "credit_card",
    "hascrcard": "credit_card",
    "has_cr_card": "credit_card",
    "is_active_member": "isactivemember",
    "estimated_salary": "estimatedsalary",
    "satisfactionscore": "satisfaction_score",
    "pointearned": "point_earned",
    "cardtype": "card_type",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading trained model artifacts...")
def load_artifacts() -> dict[str, object]:
    missing_paths = [str(path.relative_to(PROJECT_ROOT)) for path in ARTIFACT_PATHS.values() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing required model artifacts: {', '.join(missing_paths)}")

    return {name: joblib.load(path) for name, path in ARTIFACT_PATHS.items()}


@st.cache_data(show_spinner=False)
def load_sample_dataset() -> pd.DataFrame:
    if not SAMPLE_DATA_PATH.exists():
        raise FileNotFoundError("The bundled demonstration dataset was not found.")
    dataset = pd.read_csv(SAMPLE_DATA_PATH)
    return dataset.drop(columns=["Churned"], errors="ignore").head(700)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO

    return pd.read_csv(BytesIO(file_bytes))


def standardize_input_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    standardized = column_stand(dataset)
    rename_map: dict[str, str] = {}
    existing_columns = set(standardized.columns)
    for source_name, target_name in COLUMN_ALIASES.items():
        if source_name in existing_columns and target_name not in existing_columns:
            rename_map[source_name] = target_name
    return standardized.rename(columns=rename_map)


def required_feature_groups(artifacts: dict[str, object]) -> tuple[list[str], list[str]]:
    numeric_features = list(artifacts["num_imputer"].feature_names_in_)
    categorical_features = list(artifacts["cat_imputer"].feature_names_in_)
    return numeric_features, categorical_features


def display_schema(artifacts: dict[str, object]) -> None:
    numeric_features, categorical_features = required_feature_groups(artifacts)
    schema = pd.DataFrame(
        {
            "Required column": numeric_features + categorical_features,
            "Expected type": ["Numeric"] * len(numeric_features) + ["Categorical"] * len(categorical_features),
        }
    )
    st.dataframe(schema, use_container_width=True, hide_index=True)


def validate_dataset(dataset: pd.DataFrame, artifacts: dict[str, object]) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    standardized = standardize_input_columns(dataset)
    numeric_features, categorical_features = required_feature_groups(artifacts)
    required_features = numeric_features + categorical_features
    missing_features = [column for column in required_features if column not in standardized.columns]
    extra_columns = [column for column in standardized.columns if column not in required_features and column != TARGET_COLUMN]
    if missing_features:
        return None, missing_features, extra_columns
    return standardized, [], extra_columns


def prepare_features(dataset: pd.DataFrame, artifacts: dict[str, object]) -> tuple[pd.DataFrame, list[str], list[str]]:
    numeric_features, categorical_features = required_feature_groups(artifacts)
    working_data = dataset.loc[:, numeric_features + categorical_features].copy()
    conversion_warnings: list[str] = []

    numeric_data = pd.DataFrame(index=working_data.index)
    for column in numeric_features:
        original_values = working_data[column]
        converted_values = pd.to_numeric(original_values, errors="coerce")
        non_empty_values = original_values.notna() & original_values.astype("string").str.strip().ne("")
        invalid_count = int((converted_values.isna() & non_empty_values).sum())
        if invalid_count:
            conversion_warnings.append(f"{column}: {invalid_count} non-numeric value(s) were treated as missing.")
        numeric_data[column] = converted_values

    numeric_data = numeric_data.astype(float)
    numeric_data.loc[:, numeric_features] = artifacts["num_imputer"].transform(numeric_data[numeric_features])

    categorical_data = working_data.loc[:, categorical_features].copy()
    for column in categorical_features:
        categorical_data[column] = categorical_data[column].astype(object).where(categorical_data[column].notna(), np.nan)
        categorical_data[column] = categorical_data[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    categorical_data.loc[:, categorical_features] = artifacts["cat_imputer"].transform(categorical_data[categorical_features])

    unseen_category_warnings: list[str] = []
    for column, known_categories in zip(categorical_features, artifacts["encoder"].categories_):
        unknown_count = int((~categorical_data[column].isin(known_categories)).sum())
        if unknown_count:
            unseen_category_warnings.append(
                f"{column}: {unknown_count} unseen value(s) were encoded as unknown categories."
            )

    for column, (lower_bound, upper_bound) in artifacts["outlier_bounds"].items():
        if column in numeric_data:
            numeric_data[column] = numeric_data[column].clip(lower=lower_bound, upper=upper_bound)

    encoded_values = artifacts["encoder"].transform(categorical_data[categorical_features])
    if hasattr(encoded_values, "toarray"):
        encoded_values = encoded_values.toarray()
    encoded_data = pd.DataFrame(
        encoded_values,
        columns=artifacts["encoder"].get_feature_names_out(categorical_features),
        index=working_data.index,
    )

    scaler_features = list(artifacts["scaler"].feature_names_in_)
    numeric_data.loc[:, scaler_features] = artifacts["scaler"].transform(numeric_data[scaler_features])

    processed_data = pd.concat([numeric_data, encoded_data], axis=1)
    feature_order = list(artifacts["model"].feature_names_in_)
    missing_processed_features = [column for column in feature_order if column not in processed_data.columns]
    if missing_processed_features:
        raise ValueError(f"Processed data is missing model features: {', '.join(missing_processed_features)}")
    processed_data = processed_data.loc[:, feature_order]
    return processed_data, conversion_warnings, unseen_category_warnings


def build_prediction_results(dataset: pd.DataFrame, artifacts: dict[str, object]) -> tuple[pd.DataFrame, list[str], list[str]]:
    standardized_dataset = standardize_input_columns(dataset)
    processed_data, conversion_warnings, unseen_category_warnings = prepare_features(standardized_dataset, artifacts)
    model = artifacts["model"]
    predictions = model.predict(processed_data)
    churn_class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(processed_data)[:, churn_class_index]

    results = dataset.copy()
    results["Churn Prediction"] = np.where(predictions == 1, "Likely to Churn", "Not Likely to Churn")
    results["Churn Probability"] = probabilities
    results["Risk Level"] = pd.cut(
        probabilities,
        bins=[-np.inf, 0.40, 0.70, np.inf],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    ).astype(str)
    results = results.sort_values("Churn Probability", ascending=False, kind="stable").reset_index(drop=True)
    return results, conversion_warnings, unseen_category_warnings


def show_metrics(results: pd.DataFrame) -> None:
    total_customers = len(results)
    likely_to_churn = int((results["Churn Prediction"] == "Likely to Churn").sum())
    high_risk = int((results["Risk Level"] == "High Risk").sum())
    metric_columns = st.columns(6)
    metric_columns[0].metric("Total Customers", f"{total_customers:,}")
    metric_columns[1].metric("Likely to Churn", f"{likely_to_churn:,}")
    metric_columns[2].metric("Not Likely to Churn", f"{total_customers - likely_to_churn:,}")
    metric_columns[3].metric("High Risk", f"{high_risk:,}")
    metric_columns[4].metric("Average Probability", f"{results['Churn Probability'].mean():.1%}")
    metric_columns[5].metric("Predicted Churn Rate", f"{likely_to_churn / total_customers:.1%}")


def show_charts(results: pd.DataFrame) -> None:
    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Churn Prediction Distribution")
        prediction_counts = results["Churn Prediction"].value_counts().reindex(["Likely to Churn", "Not Likely to Churn"], fill_value=0)
        st.bar_chart(prediction_counts.rename("Customers"))
    with right_column:
        st.subheader("Risk-Level Distribution")
        risk_counts = results["Risk Level"].value_counts().reindex(["High Risk", "Medium Risk", "Low Risk"], fill_value=0)
        st.bar_chart(risk_counts.rename("Customers"))

    st.subheader("Churn Probability Distribution")
    probability_bins = pd.cut(
        results["Churn Probability"],
        bins=[0, 0.20, 0.40, 0.70, 1.0],
        include_lowest=True,
        labels=["0–20%", "20–40%", "40–70%", "70–100%"],
    )
    st.bar_chart(probability_bins.value_counts(sort=False).rename("Customers"))


def show_results_table(results: pd.DataFrame, key_prefix: str = "results") -> None:
    filter_options = [
        "All Customers",
        "Likely to Churn",
        "Not Likely to Churn",
        "High Risk",
        "Medium Risk",
        "Low Risk",
    ]
    selected_filter = st.selectbox("Filter customers", filter_options, key=f"{key_prefix}_filter")
    search_text = st.text_input("Search customer details", key=f"{key_prefix}_search", placeholder="Customer ID, surname, geography...")

    filtered_results = results.copy()
    if selected_filter in {"Likely to Churn", "Not Likely to Churn"}:
        filtered_results = filtered_results[filtered_results["Churn Prediction"] == selected_filter]
    elif selected_filter != "All Customers":
        filtered_results = filtered_results[filtered_results["Risk Level"] == selected_filter]

    if search_text.strip():
        search_matches = filtered_results.astype("string").apply(
            lambda column: column.str.contains(search_text.strip(), case=False, na=False)
        )
        filtered_results = filtered_results[search_matches.any(axis=1)]

    preferred_columns = [
        "CustomerId",
        "customerid",
        "Surname",
        "surname",
        "Geography",
        "geography",
        "Gender",
        "gender",
        "Age",
        "age",
        "Balance",
        "balance",
        "NumOfProducts",
        "numofproducts",
        "IsActiveMember",
        "Complain",
        "complain",
        "Churn Prediction",
        "Churn Probability",
        "Risk Level",
    ]
    visible_columns = list(dict.fromkeys(column for column in preferred_columns if column in filtered_results.columns))
    if not visible_columns:
        visible_columns = list(filtered_results.columns)
    st.caption(f"Showing {len(filtered_results):,} of {len(results):,} customers, ordered by churn probability.")
    st.dataframe(filtered_results[visible_columns], use_container_width=True, hide_index=True)


def show_customer_detail(results: pd.DataFrame) -> None:
    with st.expander("Customer detail view"):
        display_label = results.index.to_series().add(1).astype(str) + " — " + results["Churn Probability"].map(lambda value: f"{value:.1%}")
        selected_index = st.selectbox("Select a customer", options=results.index, format_func=lambda index: display_label.loc[index])
        customer = results.loc[selected_index]
        summary_columns = st.columns(3)
        summary_columns[0].metric("Prediction", customer["Churn Prediction"])
        summary_columns[1].metric("Churn Probability", f"{customer['Churn Probability']:.1%}")
        summary_columns[2].metric("Risk Level", customer["Risk Level"])
        identifier_fields = [column for column in customer.index if column.lower() in IDENTIFIER_COLUMNS]
        if identifier_fields:
            st.write("Customer identifiers")
            st.json({column: customer[column] for column in identifier_fields})
        detail_fields = [
            column
            for column in customer.index
            if column not in {"Churn Prediction", "Churn Probability", "Risk Level"}
            and column.lower() != TARGET_COLUMN
        ]
        st.dataframe(customer[detail_fields].rename("Value").to_frame(), use_container_width=True)


def show_prediction_workspace(artifacts: dict[str, object]) -> None:
    st.header("Predict Churn")
    st.write("Upload a CSV or use bundled demonstration data. Predictions use only the saved Logistic Regression model and fitted preprocessing artifacts.")
    source = st.radio("Choose a dataset", ["Upload Customer Dataset", "Try Sample Dataset"], horizontal=True)
    dataset: pd.DataFrame | None = None
    dataset_label = ""

    if source == "Upload Customer Dataset":
        uploaded_file = st.file_uploader("Upload a customer CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                dataset = load_uploaded_csv(uploaded_file.getvalue())
                dataset_label = uploaded_file.name
            except EmptyDataError:
                st.error("The uploaded CSV is empty. Please upload a CSV with at least one customer row.")
            except Exception:
                logger.exception("Unable to parse uploaded CSV")
                st.error("The file could not be read as a CSV. Check the file format and try again.")
    else:
        if st.button("Use Sample Dataset", type="secondary"):
            try:
                st.session_state.sample_dataset = load_sample_dataset()
                st.success("Bundled demonstration data is ready for prediction.")
            except Exception:
                logger.exception("Unable to load bundled sample dataset")
                st.error("The bundled demonstration dataset could not be loaded.")
        dataset = st.session_state.get("sample_dataset")
        dataset_label = "Bundled demonstration data" if dataset is not None else ""

    if dataset is None:
        st.info("Choose a dataset to preview it before generating predictions.")
        return
    if dataset.empty:
        st.error("The selected dataset has no customer rows.")
        return

    st.caption(f"Dataset: {dataset_label} · {len(dataset):,} customer row(s)")
    st.subheader("Data Preview")
    st.dataframe(dataset.head(20), use_container_width=True, hide_index=True)

    standardized_data, missing_features, extra_columns = validate_dataset(dataset, artifacts)
    if missing_features:
        st.error("Predictions cannot be generated because required columns are missing: " + ", ".join(missing_features))
        st.write("Required input schema")
        display_schema(artifacts)
        return
    if extra_columns:
        st.info("Extra columns will be preserved in results but ignored by the model: " + ", ".join(extra_columns))
    if TARGET_COLUMN in standardized_data.columns:
        st.info("The target column `Churned` is ignored and is never used as a prediction feature.")

    if st.button("Generate Predictions", type="primary"):
        try:
            results, conversion_warnings, unseen_category_warnings = build_prediction_results(dataset, artifacts)
            st.session_state.prediction_results = results
            st.session_state.prediction_dataset_label = dataset_label
            st.session_state.prediction_warnings = conversion_warnings + unseen_category_warnings
            st.success(f"Generated predictions for {len(results):,} customer(s).")
        except Exception:
            logger.exception("Prediction failed")
            st.error("Predictions could not be generated. Confirm that the CSV uses the required schema and valid data values.")

    if "prediction_results" in st.session_state:
        st.subheader("Latest Prediction Summary")
        show_metrics(st.session_state.prediction_results)


def show_dashboard() -> None:
    st.header("Dashboard")
    results = st.session_state.get("prediction_results")
    if results is None:
        st.info("Generate predictions from the Predict Churn page to view the dashboard.")
        return
    st.caption(f"Results for: {st.session_state.get('prediction_dataset_label', 'selected dataset')}")
    show_metrics(results)
    show_charts(results)


def show_customer_results() -> None:
    st.header("Customer Results")
    results = st.session_state.get("prediction_results")
    if results is None:
        st.info("Generate predictions from the Predict Churn page to explore customer results.")
        return
    for warning in st.session_state.get("prediction_warnings", []):
        st.warning(warning)
    show_results_table(results)
    show_customer_detail(results)
    download_data = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Prediction Results",
        data=download_data,
        file_name="customer_churn_predictions.csv",
        mime="text/csv",
        type="primary",
    )


def show_about(artifacts: dict[str, object]) -> None:
    st.header("About")
    st.write("This application scores customer records with the saved Logistic Regression churn model. It provides model estimates, not guarantees of future customer behavior.")
    st.subheader("Required Features")
    display_schema(artifacts)
    st.info("`Complain` is included because it represents a complaint recorded before the churn prediction point in the source dataset.")
    st.caption("Risk labels are application-defined thresholds: High Risk ≥ 70%, Medium Risk ≥ 40%, and Low Risk < 40%.")


def main() -> None:
    st.set_page_config(page_title="Customer Churn Prediction System", page_icon="📈", layout="wide")
    st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #e6eaf0;
        border-radius: 10px;
        padding: 0.75rem;
        color: #111827 !important;
    }

    [data-testid="stMetric"] *,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: #111827 !important;
    }

    [data-testid="stMetricLabel"] {
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.title("Customer Churn Prediction System")
    st.caption("Predict customers who are at risk of churning using machine learning.")

    try:
        artifacts = load_artifacts()
    except Exception:
        logger.exception("Model artifacts could not be loaded")
        st.error("The saved model or preprocessing artifacts are unavailable. Check the project models directory before running predictions.")
        return

    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to", ["🏠 Dashboard", "📤 Predict Churn", "👥 Customer Results", "ℹ️ About"])
        st.divider()
        st.subheader("About the Model")
        st.write("**Model:** Logistic Regression")
        st.write("**Task:** Customer churn prediction")
        st.write("**Output:** Churn probability, prediction, and risk level")
        st.divider()
        st.subheader("Prediction Instructions")
        st.write("Upload a CSV containing every required feature, preview it, then select Generate Predictions.")
        st.write("`Complain` is a valid pre-prediction feature in this dataset.")

    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📤 Predict Churn":
        show_prediction_workspace(artifacts)
    elif page == "👥 Customer Results":
        show_customer_results()
    else:
        show_about(artifacts)


if __name__ == "__main__":
    main()
