
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
df = df.drop(
    columns=[
        "RowNumber",
        "CustomerId",
        "Surname",
        "Complain"
    ]
)
EXPERIMENTAL_CATEGORICAL_FEATURES = [
    feature for feature in CATEGORICAL_FEATURES
    if feature != "complain"
]

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
    df = column_stand(df)
    df = remove_dup(df)
    "filling null values"
    df,num_imputer = fill_num(df)
    df,cat_imputer = fill_cat(df)
    "removing outliers"
    df = remove_outlier(df)
    "encoding"
    df,encoder = one_hot(df,EXPERIMENTAL_CATEGORICAL_FEATURES)
    "scaling"
    df,standard_model = stand_scaler(
        df,["creditscore","tenure","estimatedsalary",
            "satisfaction_score","point_earned","balance"]
        )
    df,robust_model = robust_scaler(
        df,
        ["age","numofproducts"]
        )
    preprocessors = {
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "encoder": encoder,
        "standard_scaler": standard_model,
        "robust_scaler": robust_model
    }
    return df,preprocessors
    
"using saved models of X_train in X_test"

def preprocess_Xtest(df,preprocessors):
    df = column_stand(df)
    "filling null values"
    df,_ = fill_num(
        df,
        imputer = preprocessors["num_imputer"]
          )
    df,_ = fill_cat(
        df,
        imputer = preprocessors["cat_imputer"]
        )
    "encoding"
    df,_ = one_hot(
        df,
        EXPERIMENTAL_CATEGORICAL_FEATURES,
        encoder = preprocessors["encoder"]
        )
    "scaling"
    df,_ = stand_scaler(
        df,["creditscore","tenure","estimatedsalary",
            "satisfaction_score","point_earned","balance"],
            preprocessors["standard_scaler"]
        )
    df,_ = robust_scaler(
        df,
        ["age","numofproducts"],
        preprocessors["robust_scaler"]
        )
  

    return df
#data_preprocessing
X_train,preprocessors = preprocess_Xtrain(X_train)
X_test = preprocess_Xtest(X_test,preprocessors)



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
print(results)
"""preprocessors = {
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "encoder": encoder,
        "standard_scaler": standard_model,
        "robust_scaler": robust_model
    }

best_model = trained_models["Gradient Boosting"]

num_imputer = preprocessors["num_imputer"]
cat_imputer = preprocessors["cat_imputer"]
encoder = preprocessors["encoder"]
Standard_scaler = preprocessors["standard_scaler"]
robust_scaler_ = preprocessors["robust_scaler"]

joblib.dump(best_model,"data/best_model.pkl")
joblib.dump(num_imputer,"data/num_imputer.pkl")
joblib.dump(cat_imputer,"data/cat_imputer.pkl")
joblib.dump(encoder,"data/encoder.pkl")
joblib.dump(Standard_scaler,"data/Standard_scaler.pkl")
joblib.dump(robust_scaler_,"data/robust_scaler_.pkl")"""


"---------------------------"
"Random forest , Gradient boosting , XGboost are top ranking models in evalution phase ,"
"after tuning these models their results  becomed more worse than in evaluation phase."
"So , we choosed (Gradient Boosting) as best model because it has given beat result among all models in evalution phase."
"---------------------------"
#model_tuning_
""""_______random_forest_tuning________"
param_grid = rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}
rf_result = gridsearch(model = trained_models["Random Forest"],
                       param_grid = rf_param_grid,
                       X_train = X_train,
                       y_train = y_train
                       )
"___random_forest___"
best_rf = rf_result["best_model"]

"_____Gradient Boost tunning______"
gb_param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

gb_results = gridsearch(model = trained_models["Gradient Boosting"],
                     param_grid=gb_param_grid,
                     X_train = X_train,
                     y_train = y_train
                     )
best_gb = gb_results["best_model"]

"______XG_boost_tuning_____"
xgb_param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 9,11],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.3, 0.5],
    "min_child_weight": [1, 3, 5]
}

XG_results = randomized_search_tuning(model = trained_models["XGboost"],
                                      param_distributions = xgb_param_dist,
                                      X_train = X_train,
                                      y_train = y_train
                                      )
best_xg = XG_results["best_model"]


"____cross__val_________"
best_model = best_rf
score = cross_val_score(
        best_model,
        X_train,
        y_train,
        cv=5,
       scoring="f1"
)
print("rf:",score.mean())
best_model = best_gb
score = cross_val_score(
        best_model,
        X_train,
        y_train,
        cv=5,
       scoring="f1"
)
print("gb:",score.mean())

best_model = best_xg
score = cross_val_score(
        best_model,
        X_train,
        y_train,
        cv=5,
       scoring="f1"
)
print("xg:",score.mean())"""
