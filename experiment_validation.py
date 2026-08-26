"""Leakage-safe validation experiments for the customer churn project.

Run from the project root: ``python experiment_validation.py``.
This is intentionally separate from ``src/main.py`` and does not persist models.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.preprocessing import column_stand, remove_dup


RANDOM_STATE = 42
TARGET = "churned"
COMPLAIN = "complain"
CONTINUOUS_FEATURES = [
    "balance", "estimatedsalary", "creditscore", "age", "point_earned", "tenure"
]
CATEGORICAL_FEATURES = ["geography", "gender", "card_type"]
DROP_COLUMNS = ["rownumber", "customerid", "surname"]


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip each numeric column to IQR bounds learned only during ``fit``."""

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(values, [25, 75], axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - 1.5 * iqr
        self.upper_bounds_ = q3 + 1.5 * iqr
        return self

    def transform(self, X):
        return np.clip(
            np.asarray(X, dtype=float), self.lower_bounds_, self.upper_bounds_
        )


def make_preprocessor(features):
    """Mirror the project's preprocessing philosophy with fold-safe estimators."""
    continuous = [column for column in CONTINUOUS_FEATURES if column in features]
    categorical = [column for column in CATEGORICAL_FEATURES if column in features]
    remaining_numeric = [
        column
        for column in features
        if column not in continuous + categorical
        and pd.api.types.is_numeric_dtype(features[column])
    ]

    transformers = []
    if continuous:
        transformers.append((
            "continuous",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("iqr_clipper", IQRClipper()),
                ("scaler", StandardScaler()),
            ]),
            continuous,
        ))
    if remaining_numeric:
        transformers.append((
            "other_numeric",
            SimpleImputer(strategy="median"),
            remaining_numeric,
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(random_state=RANDOM_STATE),
    }


SCORING = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}
DISPLAY_NAMES = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "f1": "F1", "roc_auc": "ROC-AUC",
}


def run_cv(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, model in models().items():
        pipeline = Pipeline([
            ("preprocessing", make_preprocessor(X)),
            ("model", model),
        ])
        scores = cross_validate(
            pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=1, error_score="raise"
        )
        results[name] = {
            metric: (scores[f"test_{metric}"].mean(), scores[f"test_{metric}"].std())
            for metric in SCORING
        }
    return results


def print_results(title, results):
    print(f"\n================ {title} ================")
    for name, metrics in results.items():
        print(f"\n{name}")
        for metric in SCORING:
            mean, std = metrics[metric]
            print(f"{DISPLAY_NAMES[metric]:<9}: {mean:.4f} ± {std:.4f}")


def comparison_frame(without, with_complain):
    rows = []
    for version, result_set in (("Without Complain", without), ("With Complain", with_complain)):
        for model, metrics in result_set.items():
            rows.append({
                "Model": model,
                "Version": version,
                **{DISPLAY_NAMES[key]: metrics[key][0] for key in SCORING},
            })
    return pd.DataFrame(rows)


def shortlist(results):
    # Ordered by the requested priorities; standard deviation breaks otherwise equal means.
    def rank(item):
        _, metrics = item
        ordered = ("f1", "recall", "roc_auc", "precision", "accuracy")
        return tuple(-metrics[key][0] for key in ordered) + tuple(metrics[key][1] for key in ordered)
    return [name for name, _ in sorted(results.items(), key=rank)[:3]]


def main():
    path = Path("data/raw/customer churn.csv")
    raw_df = pd.read_csv(path)

    print("================ COMPLAIN VS CHURNED ================")
    print("\nCorrelation")
    print(raw_df[["Complain", "Churned"]].corr())
    print("\nRaw contingency table")
    print(pd.crosstab(raw_df["Complain"], raw_df["Churned"]))
    print("\nRow-wise percentages")
    rates = pd.crosstab(raw_df["Complain"], raw_df["Churned"], normalize="index")
    print(rates)
    churn_rates = rates.get(1, pd.Series(dtype=float))
    for value in (0, 1):
        print(f"Complain = {value}: Churn rate = {churn_rates.get(value, np.nan):.2%}")
    print(f"Difference in churn rate = {(churn_rates.get(1, np.nan) - churn_rates.get(0, np.nan)):.2%}")
    print("This establishes a statistical relationship only; it does not by itself prove leakage.")

    # Reuse the project's standardization and duplicate-removal helpers before splitting.
    df = remove_dup(column_stand(raw_df))
    df = df.drop(columns=DROP_COLUMNS)
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nHeld-out final test set: 20% ({len(df) - len(X_train)} rows), not used in CV.")
    print(f"Cross-validation training portion: {len(X_train)} rows.")

    without = run_cv(X_train.drop(columns=COMPLAIN), y_train)
    with_complain = run_cv(X_train, y_train)
    print_results("WITHOUT COMPLAIN", without)
    print_results("WITH COMPLAIN", with_complain)

    comparison = comparison_frame(without, with_complain)
    print("\n================ DIRECT COMPARISON ================")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    for metric in ("ROC-AUC", "F1", "Recall"):
        print(f"With Complain {metric} - Without Complain {metric} (by model):")
        for model in without:
            difference = with_complain[model][metric.lower().replace("-", "_")][0] - without[model][metric.lower().replace("-", "_")][0]
            print(f"  {model}: {difference:+.4f}")

    without_shortlist, with_shortlist = shortlist(without), shortlist(with_complain)
    print("\n================ MODEL SHORTLIST ================")
    print("\nWITHOUT COMPLAIN")
    for index, name in enumerate(without_shortlist, 1): print(f"{index}. {name}")
    print("\nWITH COMPLAIN")
    for index, name in enumerate(with_shortlist, 1): print(f"{index}. {name}")

    suspicious = any(
        metrics["roc_auc"][0] >= .99 and metrics["f1"][0] >= .99 and metrics["recall"][0] >= .99
        for metrics in with_complain.values()
    )
    print("\n====================================================")
    print("FINAL EXPERIMENT SUMMARY")
    print("====================================================")
    print("1. Complain vs Churn relationship: reported above from the raw data.")
    print("2. CV results WITHOUT Complain: reported above using the untouched-test-set protocol.")
    print("3. CV results WITH Complain: reported above using identical fold-safe preprocessing.")
    print("4. Direct comparison: reported above; differences are paired by model family.")
    print("5. Top candidate models: listed above; no final deployment model has been selected.")
    if suspicious:
        print("6. The With Complain result is extremely high and requires leakage/timing investigation.")
    else:
        print("6. Complain still requires timing validation before deployment because availability may differ at prediction time.")
    print("The feature has an extremely strong predictive relationship with churn. Its use for deployment depends on whether Complain is available before the churn prediction timestamp.")
    print("No hyperparameter tuning was performed and src/main.py was not modified.")


if __name__ == "__main__":
    main()
