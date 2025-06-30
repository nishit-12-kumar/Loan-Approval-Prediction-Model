import os
import sys

import numpy as np
import pandas as pd
import dill

from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# AS IT IS RAHNE DENA HAIN
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys) 


# THODA CHANGE HUAA ISME...
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        best_model_overall = None
        best_score = -1

        for name, model in models.items():
            print(f"Tuning {name}...")
            if params.get(name):
                gs = GridSearchCV(model, params[name], cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            y_pred = best_model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            report[name] = score

            if score > best_score:
                best_score = score
                best_model_overall = best_model
        
        print(f"Returning best fitted model: {type(best_model_overall).__name__}")
        return report, best_model_overall

    except Exception as e:
        raise CustomException(e, sys)



# AS IT IS RAHNE DENA HAIN
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)