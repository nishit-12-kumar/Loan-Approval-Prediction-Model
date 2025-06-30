# import os
# import sys
# from dataclasses import dataclass

# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import GridSearchCV

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB


# from src.exception import CustomException
# from src.logger import logging
# from src.utils import save_object, evaluate_models


# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()

#     def initiate_model_trainer(self, train_array, test_array):
#         try:
#             logging.info("Splitting training and test data into input and target...")

#             X_train = train_array[:, :-1]
#             y_train = train_array[:, -1]

#             X_test = test_array[:, :-1]
#             y_test = test_array[:, -1]

#             # Continue with existing logic...
#             models = {
#                 "Random Forest": RandomForestClassifier(),
#                 "Decision Tree": DecisionTreeClassifier(),
#                 "KNN": KNeighborsClassifier(),
#                 "SVM": SVC(probability=True),
#                 "Logistic Regression": LogisticRegression(),
#                 "Naive Bayes": GaussianNB()
#             }

#             params = {
#                 "Random Forest": {
#                     "n_estimators": [100, 200],
#                     "max_depth": [None, 10, 20],
#                     "min_samples_split": [2, 5],
#                 },
#                 "Decision Tree": {
#                     "max_depth": [None, 10, 20],
#                     "criterion": ["gini", "entropy"]
#                 },
#                 "KNN": {
#                     "n_neighbors": [3, 5, 7],
#                     "weights": ["uniform", "distance"]
#                 },
#                 "SVM": {
#                     "C": [0.1, 1, 10],
#                     "kernel": ["linear", "rbf", "poly"],
#                     "gamma": ["scale", "auto"]
#                 },
#                 "Logistic Regression": {
#                     "C": [0.1, 1.0, 10.0],
#                     "solver": ["liblinear"]
#                 },
#                 "Naive Bayes": {}  # No tuning — default is usually optimal
#             }

#             # model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
#             model_report, best_model = evaluate_models(X_train, y_train, X_test, y_test, models, params)


#             logging.info(f"Model scores: {model_report}")

#             # Find best model
#             best_model_name = max(model_report, key=model_report.get)            
#             best_model_score = model_report[best_model_name]
#             best_model = models[best_model_name]

#             logging.info(f"Best model: {best_model_name} with accuracy: {best_model_score}")

#             logging.info("Saving best model...")

#             # Save the best model
#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )

#             from sklearn.utils.validation import check_is_fitted

#             try:
#                 check_is_fitted(best_model)
#                 print("BEST MODEL IS FITTED")
#             except:
#                 print("BEST MODEL IS NOT FITTED")

#             return best_model, best_model_score

#         except Exception as e:
#             logging.error("Error in model training")
#             raise CustomException(e, sys)















import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted

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
            try:
                check_is_fitted(best_model)
                logging.info("Confirmed: Best model is fitted.")
            except Exception as e:
                logging.error("Best model is NOT fitted.")
                raise e

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Model saved successfully.")
            return best_model, model_report

        except Exception as e:
            raise CustomException(e, sys)


