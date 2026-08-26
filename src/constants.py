# constants.py
# src/constants.py
"In preprocessing phase we have scaled column names for better reliability so , we have to also save scaled version of dataset column names."

TARGET = "churned"

CONTINUOUS_FEATURES = [
    "balance",
    "estimatedsalary",
    "creditscore",
    "age",
    "point_earned",
    "tenure"
]

DISCRETE_FEATURES = [
    "numofproducts",
    "credit_card",
    "isactivemember",
    "complain",
    "satisfaction_score"
]

CATEGORICAL_FEATURES = [
    "geography",
    "gender",
    "card_type"
]
xgb_param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 9],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.3, 0.5],
    "min_child_weight": [1, 3, 5]
}

gb_param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

param_grid = rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

"best_param_after_tuning_"
{'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300}
"""__gb___"""
{'learning_rate': 0.1, 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
"""___XG___"""
{'subsample': 0.6, 'n_estimators': 300, 'min_child_weight': 1, 'max_depth': 3, 'learning_rate': 0.05, 'gamma': 0.1, 'colsample_bytree': 1.0}

cross_val_results_ = {"rf": 0.5619682900468247,"gb": 0.586988773172576,"xg": 0.5837390946258699}