from src.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    di = DataIngestion()
    train_dataset_path, test_dataset_path = di.initiate_data_ingestion()