import pandas as pd
from src.utils import load_pickle
import os


class Prediction:
    def __init__(self):
        pass
    def Predict(self,features):
        preprocessing_path=os.path.join("artifacts/pickle","data_transformation.pkl")
        model_path=os.path.join("artifacts/pickle","model.pkl")
        preprocessing=load_pickle(preprocessing_path)
        model=load_pickle(model_path)
        processed_data=preprocessing.transform(features)
        op=model.predict(processed_data)
        return op

class GetFeature:
    def __init__(self,age,gender,smoking,yellow_finger,anxiety,peer_pressure,chronic_disease,
                 fatigue,allergy,wheezing,alcohol_consuming,coughing,shortness_of_breath,
                 swallowing_difficulty,chest_pain):
        self.age=age,
        self.gender=gender
        self.smoking=smoking
        self.yellow_finger=yellow_finger
        self.anxiety=anxiety
        self.peer_pressure=peer_pressure
        self.chronic_disease=chronic_disease
        self.fatigue=fatigue
        self.allergy=allergy
        self.wheezing=wheezing
        self.alcohol_consuming=alcohol_consuming
        self.coughing=coughing
        self.shortness_of_breath=shortness_of_breath
        self.swallowing_difficulty=swallowing_difficulty
        self.chest_pain=chest_pain


    def to_dataframe(self):
        feature={
            "age":[self.age],
            "gender":[self.gender],
            "smoking":[self.smoking],
            "yellow_finger":[self.yellow_finger],
            "anxiety":[self.anxiety],
            "peer_pressure":[self.peer_pressure],
            "chronic_disease":[self.chronic_disease],
            "fatigue":[self.fatigue],
            "allergy":[self.allergy],
            "wheezing":[self.wheezing],
            "alcohol_consuming":[self.alcohol_consuming],
            "coughing":[self.coughing],
            "shortness_of_breath":[self.shortness_of_breath],
            "swallowing_difficulty":[self.swallowing_difficulty],
            "chest_pain":[self.chest_pain]
        }
        feature=pd.DataFrame(feature)
        return feature











#%%
