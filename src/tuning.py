import pandas as pd 
import numpy as np 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


def gridsearch(model,param_grid,X_train,y_train,
               scoring = "f1",cv = 5,n_jobs = -1):
        grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs
        )
       
        grid.fit(X_train,y_train)
        return {
        "best_model": grid.best_estimator_,
        "best_params": grid.best_params_,
        "best_score": grid.best_score_,
        "cv_results": grid.cv_results_
    }

def randomized_search_tuning(model, param_distributions,X_train, y_train,
                             n_iter=100,scoring="f1",cv=5,
                             random_state=42,n_jobs=-1):

        random = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs
        )

        random.fit(X_train, y_train)

        return {
        "best_model": random.best_estimator_,
        "best_params": random.best_params_,
        "best_score": random.best_score_,
        "cv_results": random.cv_results_
        }

