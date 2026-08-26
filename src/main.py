"""Train and evaluate churn-prediction models.

Run from the project root with either ``python -m src.main`` or
``python src/main.py``.
"""

from pathlib import Path
import sys

# Support direct execution (``python src/main.py``) as well as module execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd 
import numpy as np 
from src.statistics import *
from src.preprocessing import *
from src.constants import *
from src.data_loader import *
from src.train import *
from src.evaluate import *
from src.tuning import *
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
df =load_data("data/raw/customer churn.csv")
df = df.copy()
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
"________Statistics________"
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
"""report = statistics_summary(df)
print(report.shape, end="\n\n")
print(report.summary, end="\n\n")
print(report.describe, end="\n\n")
print(report.correlation, end="\n\n")
print(report.info, end="\n\n")
"_______preprocessing________"
"___analyzing_dataet___"
analyzed = analyze(df)
print("/n")
print(analyzed.categorical_features, end="\n\n")
print(analyzed.numeric_features, end="\n\n")
print(analyzed.missing_valuecount, end="\n\n")
print(analyzed.outliers_detection, end="\n\n")"""
"_____Data_spliting____"

# The source CSV has title-cased column names at this point.  They are converted
# to lowercase later inside the preprocessing functions.

X = df.drop(columns=["Churned"])
y = df["Churned"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state = 42,
    stratify = y 
)


"-----Preprosessd_datasets of X_train and X_test----"
def preprocess_Xtrain(df):

    "standralize columns"
    df = column_stand(df)

    "remove duplicates"
    df = remove_dup(df)

    "filling null values"
    df,num_imputer = fill_num(df)
    df,cat_imputer = fill_cat(df)

    "removing outliers"
    df,outlier_bounds = remove_outlier(df,CONTINUOUS_FEATURES)

    "encoding"
    df,encoder = one_hot(df,CATEGORICAL_FEATURES)

    "scaling"
    df,standard_model = stand_scaler(
        df,CONTINUOUS_FEATURES
        )

    preprocessors = {
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "encoder": encoder,
        "standard_scaler": standard_model,
        "outlier_bounds": outlier_bounds
    }
    return df,preprocessors
    
"using saved models of X_train in X_test"

def preprocess_Xtest(df,preprocessors):

    "standralize columns"
    df = column_stand(df)

    "remove duplicates"
    df = remove_dup(df)

    "filling null values"
    df,_ = fill_num(
        df,
        imputer = preprocessors["num_imputer"]
          )
    df,_ = fill_cat(
        df,
        imputer = preprocessors["cat_imputer"]
        )
    "outlier removal"
    df,_= remove_outlier(df,
                          CONTINUOUS_FEATURES,
                          preprocessors["outlier_bounds"]
                          )
    "encoding"
    df,_ = one_hot(
        df,
        CATEGORICAL_FEATURES,
        encoder = preprocessors["encoder"]
        )
    "scaling"
    df,_ = stand_scaler(
        df,
        CONTINUOUS_FEATURES,
        preprocessors["standard_scaler"]
        )
    return df
#data_preprocessing
X_train,preprocessors = preprocess_Xtrain(X_train)
X_test = preprocess_Xtest(X_test,preprocessors)

#model_training
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "XGboost": XGBClassifier(random_state=42)
}
trained_models = train(models,X_train,y_train)

#model_evaluation_
results = evaluate_models(trained_models,X_test,y_test)

#_____________________________________________________#

best_model = trained_models["Logistic Regression"]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

num_imputer = preprocessors["num_imputer"]
cat_imputer = preprocessors["cat_imputer"]
encoder = preprocessors["encoder"]
Standard_scaler = preprocessors["standard_scaler"]
outlier_bonds = preprocessors["outlier_bounds"]

joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
joblib.dump(num_imputer, MODEL_DIR / "num_imputer.pkl")
joblib.dump(cat_imputer, MODEL_DIR / "cat_imputer.pkl")
joblib.dump(encoder, MODEL_DIR / "encoder.pkl")
joblib.dump(Standard_scaler, MODEL_DIR / "Standard_scaler.pkl")
joblib.dump(outlier_bonds, MODEL_DIR / "outlier_bonds.pkl")


