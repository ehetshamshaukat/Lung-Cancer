import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
import os
from src.utils import save_file_as_pickle
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

pd.set_option("display.max_columns",500)
@dataclass
class DataTransformationConfig:
    data_transformation_pickle_file_path = os.path.join("artifacts/pickle", "data_transformation.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_pickle = DataTransformationConfig()

    def transform_data(self):
        numerical_columns = ["age"]
        categorical_columns = ['gender', 'smoking', 'yellow_fingers', 'anxiety', 'peer_pressure', "chronic_disease",
                               'fatigue', 'allergy', 'wheezing', 'alcohol_consuming', 'coughing',
                               'shortness_of_breath', 'swallowing_difficulty', 'chest_pain']

        numerical_column_pipeline = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("standardscaler", StandardScaler())
        ])
        categorical_column_pipeline = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehotencoder", OneHotEncoder(sparse_output=False)),
            ("standardscaler", StandardScaler())
        ])

        processing = ColumnTransformer([
            ("numerical_column_pipeline", numerical_column_pipeline, numerical_columns),
            ("categorical_column_pipeline", categorical_column_pipeline, categorical_columns)
        ])
        return processing

    def initiate_data_transformation(self, train_dataset_path, test_dataset_path):
        try:
            # Loading train and test dataset
            train_dataset = pd.read_csv(train_dataset_path)
            test_dataset = pd.read_csv(test_dataset_path)

            data_transform = self.transform_data()

            # changing column name into lower case
            train_dataset.columns = train_dataset.columns.str.lower()
            test_dataset.columns = test_dataset.columns.str.lower()

            train_dataset.columns = train_dataset.columns.str.rstrip(" ")
            test_dataset.columns = test_dataset.columns.str.rstrip(" ")

            train_dataset.columns = train_dataset.columns.str.replace(" ", "_")
            test_dataset.columns = test_dataset.columns.str.replace(" ", "_")
            print(train_dataset.columns)
            print(test_dataset.columns)

            # replacing input value of each column
            columns = ['gender', 'smoking', 'yellow_fingers', 'anxiety', 'peer_pressure', 'chronic_disease',
                       'fatigue', 'allergy', 'wheezing', 'alcohol_consuming', 'coughing', 'shortness_of_breath',
                       'swallowing_difficulty', 'chest_pain']
            for col in columns:
                train_dataset[col].replace([1, 2], ["no", "yes"], inplace=True)
                test_dataset[col].replace([1, 2], ["no", "yes"], inplace=True)

            train_dataset["lung_cancer"].replace(["NO", "YES"], [0, 1], inplace=True)
            test_dataset["lung_cancer"].replace(["NO", "YES"], [0, 1], inplace=True)

            # splitting depended and independed features
            target_column = "lung_cancer"
            column_to_drop = "lung_cancer"

            xtrain = train_dataset.drop(columns=column_to_drop, axis=1)
            ytrain = train_dataset[target_column]

            xtest = test_dataset.drop(columns=column_to_drop, axis=1)
            ytest = test_dataset[target_column]

            # data transformation
            xtrain_transform_data = data_transform.fit_transform(xtrain)
            xtest_transform_data = data_transform.transform(xtest)

            # concatination depended and independed features
            train_transform_dataset = np.c_[xtrain_transform_data, np.array(ytrain)]
            test_transform_dataset = np.c_[xtest_transform_data, np.array(ytest)]

            # saving data transformation in pickle format
            save_file_as_pickle(data_transform, self.data_transformation_pickle.data_transformation_pickle_file_path)
            # returning transformed dataset
            return (train_transform_dataset, test_transform_dataset)

        except Exception as e:
            raise e
