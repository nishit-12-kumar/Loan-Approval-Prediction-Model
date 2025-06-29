import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts' , "preprocessor.pkl" )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train dataset...")
            df = pd.read_csv(train_path)

            # Handle missing values
            categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed',
                                'Loan_Amount_Term', 'Credit_History']

            for col in categorical_cols:
                df[col].fillna(df[col].mode()[0], inplace=True)

            df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

            # Drop Loan_ID if exists
            if 'Loan_ID' in df.columns:
                df.drop('Loan_ID', axis=1, inplace=True)

            # Encode categorical variables
            encode_cols = ['Gender', 'Married', 'Dependents', 'Education',
                           'Self_Employed', 'Property_Area']

            le = LabelEncoder()
            for col in encode_cols:
                df[col] = le.fit_transform(df[col])

            # Map target
            df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

            # Split into features and target
            X = df.drop('Loan_Status', axis=1)
            y = df['Loan_Status']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            # Feature scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Combine scaled features with target
            train_arr = np.c_[X_train_scaled, y_train]
            test_arr = np.c_[X_test_scaled, y_test]

            # Save the preprocessor
            save_object(self.data_transformation_config.preprocessor_obj_file_path, scaler)

            logging.info("Data transformation completed.")
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.error("Error during data transformation")
            raise CustomException(e, sys)




