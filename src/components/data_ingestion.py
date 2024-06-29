import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os

@dataclass
class DataIngestionConfig:
    train_dataset_path=os.path.join("artifacts/train_test_dataset","train_dataset.csv")
    test_dataset_path=os.path.join("artifacts/train_test_dataset","test_dataset.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # reading dataset
            df=pd.read_csv("dataset/lung cancer.csv")
            # creating directory to store train and test dataset
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_dataset_path),exist_ok=True)
            # splitting train and test dataset
            train_dataset,test_dataset=train_test_split(df,test_size=0.3,random_state=69)
            # saving train and test dataset
            train_dataset.to_csv(self.data_ingestion_config.train_dataset_path,header=True,index=False)
            test_dataset.to_csv(self.data_ingestion_config.test_dataset_path,header=True,index=False)
            # returning train and test dataset path for further use
            return (self.data_ingestion_config.train_dataset_path,self.data_ingestion_config.test_dataset_path)
        except Exception as e:
            raise e