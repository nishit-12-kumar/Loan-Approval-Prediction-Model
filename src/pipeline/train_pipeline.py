import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException



if __name__ == "__main__":
    try:
        logging.info("Starting training pipeline...")

        # Step 1: Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        logging.info(f" Ingested files:\nTrain: {train_path}\nTest: {test_path}")

        # Step 2: Data Transformation
        transformer = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(train_path, test_path)
        logging.info(f" Data transformed. Preprocessor saved at: {preprocessor_path}")

        # Step 3: Model Training
        trainer = ModelTrainer()
        model, score = trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f" Model training complete. Accuracy: {score:.4f}")

        print(f" Final model accuracy: {score:.4f}")

    except Exception as e:
        raise CustomException(e, sys)
