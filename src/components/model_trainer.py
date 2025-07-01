import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting data into features and target...")
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "KNN": KNeighborsClassifier(),
                "SVM": SVC(probability=True),
                "Logistic Regression": LogisticRegression(),
                "Naive Bayes": GaussianNB()
            }

            params = {
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10],
                    "min_samples_split": [2, 5]
                },
                "Decision Tree": {
                    "max_depth": [None, 10],
                    "criterion": ["gini", "entropy"]
                },
                "KNN": {
                    "n_neighbors": [3, 5],
                    "weights": ["uniform", "distance"]
                },
                "SVM": {
                    "C": [1, 10],
                    "kernel": ["rbf", "linear"],
                    "gamma": ["scale"]
                },
                "Logistic Regression": {
                    "C": [1.0, 10.0],
                    "solver": ["liblinear"]
                },
                "Naive Bayes": {}  # No params
            }

            logging.info("Running GridSearchCV on all models...")
            model_report, best_model = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            # Confirm model is fitted before saving
            # try:
            #     check_is_fitted(best_model)
            #     logging.info("Confirmed: Best model is fitted.")
            # except Exception as e:
            #     logging.error("Best model is NOT fitted.")
            #     raise e

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Model saved successfully.")
            return best_model, model_report

        except Exception as e:
            raise CustomException(e, sys)


