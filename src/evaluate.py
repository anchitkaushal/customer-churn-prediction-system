import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def evaluate_models(models, X_test, y_test):
    result = []
    for name,model in models.items():

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)

        """print(f"\n{name}")
        print(cm)"""
        result.append({
           "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "roc_auc_score": roc_auc_score(y_test,y_prob)
        })

    return pd.DataFrame(result)