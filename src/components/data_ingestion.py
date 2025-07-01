import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


# AS IT IS COPY PASTE...
@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")



class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process...") 

        try:

            # Load the Dataset
            df = pd.read_csv('Notebook\data\LoanApprovalPrediction.csv')
            logging.info("Dataset loaded successfully.")

            # Drop Loan_ID column
            if 'Loan_ID' in df.columns:
                df.drop(columns=["Loan_ID"], inplace=True)
                logging.info("Dropped Loan_ID column.")


            # ***************************** ITNA COMMON RAHEGA LAGBHAG SARRE PROJECTS ME.... *****************************

            # Create artifacts directory if not exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Saved raw dataset.")

            # Train-test split
            logging.info("Train test split initiated")
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Performed train-test split.")

            # Save train and test data
            train_df.to_csv(self.ingestion_config.train_data_path, index=False)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Saved train and test datasets.")

            logging.info("Ingestion of the data is Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Error occurred during data ingestion.")
            raise CustomException(e,sys)
        

# Only Used to run data_ingestion.py file ,seprately ==> To check whether there is no Error...

# if __name__ == "__main__":
#     obj = DataIngestion()
#     train_data_path, test_data_path = obj.initiate_data_ingestion()

#     print("Train data at:", train_data_path)
#     print("Test data at:", test_data_path)