from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining

if __name__ == '__main__':
    di=DataIngestion()
    train_dataset_path,test_dataset_path=di.initiate_data_ingestion()
    dt=DataTransformation()
    transform_train_data,transform_test_data=dt.initiate_data_transformation(train_dataset_path,test_dataset_path)
    mt=ModelTraining()
    mt.initiate_model_training(transform_train_data,transform_test_data)