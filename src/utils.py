import os
import sys

import numpy as np
import pandas as pd
import dill

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
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name in models:
            model = models[model_name]
            params = param[model_name]

            # Tune model using GridSearchCV
            gs = GridSearchCV(model, params, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)

            # Set best parameters
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict and calculate accuracy
            y_test_pred = model.predict(X_test)
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)



# AS IT IS RAHNE DENA HAIN
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)