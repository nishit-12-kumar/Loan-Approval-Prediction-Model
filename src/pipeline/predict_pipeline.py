import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            # Step 1: Encode categorical fields manually
            features_encoded = features.copy()

            encoding_maps = {
                "Gender": {"Male": 1, "Female": 0},
                "Married": {"Yes": 1, "No": 0},
                "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
                "Education": {"Graduate": 1, "Not Graduate": 0},
                "Self_Employed": {"Yes": 1, "No": 0},
                "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0}
            }

            for col, mapping in encoding_maps.items():
                features_encoded[col] = features_encoded[col].map(mapping)

            # Step 2: Scale the numeric features
            data_scaled = preprocessor.transform(features_encoded)

            # Step 3: Predict
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        Gender: str,
        Married: str,
        Dependents: str,
        Education: str,
        Self_Employed: str,
        ApplicantIncome: float,
        CoapplicantIncome: float,
        LoanAmount: float,
        Loan_Amount_Term: float,
        Credit_History: float,
        Property_Area: str,
    ):
        self.Gender = Gender
        self.Married = Married
        self.Dependents = Dependents
        self.Education = Education
        self.Self_Employed = Self_Employed
        self.ApplicantIncome = ApplicantIncome
        self.CoapplicantIncome = CoapplicantIncome
        self.LoanAmount = LoanAmount
        self.Loan_Amount_Term = Loan_Amount_Term
        self.Credit_History = Credit_History
        self.Property_Area = Property_Area


    def get_data_as_dataframe(self):
        try:
            data_dict = {
                "Gender": [self.Gender],
                "Married": [self.Married],
                "Dependents": [self.Dependents],
                "Education": [self.Education],
                "Self_Employed": [self.Self_Employed],
                "ApplicantIncome": [self.ApplicantIncome],
                "CoapplicantIncome": [self.CoapplicantIncome],
                "LoanAmount": [self.LoanAmount],
                "Loan_Amount_Term": [self.Loan_Amount_Term],
                "Credit_History": [self.Credit_History],
                "Property_Area": [self.Property_Area],
            }

            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e, sys)
