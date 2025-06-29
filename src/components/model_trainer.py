import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data into input and target...")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Continue with existing logic...
            models = {
                "Random Forest": RandomForestClassifier(random_state=42)
            }

            params = {
                "Random Forest": {
                    'n_estimators': [100, 150, 200],
                    'max_depth': [4, 6, 8, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            logging.info(f"Model scores: {model_report}")

            # Find best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with accuracy: {best_model_score}")

            # Retrain best model with best params
            # We'll reuse GridSearchCV here to get the best_estimator
            gs = GridSearchCV(
                estimator=RandomForestClassifier(random_state=42),
                param_grid=params["Random Forest"],
                cv=3,
                n_jobs=-1,
                verbose=1
            )
            gs.fit(X_train, y_train)
            final_model = gs.best_estimator_

            logging.info("Saving best model...")
            save_object(self.config.trained_model_file_path, final_model)

            return final_model, best_model_score

        except Exception as e:
            logging.error("Error in model training")
            raise CustomException(e, sys)
