import pandas as pd
from src.utils import load_pickle
import os


class Prediction:
    def __init__(self):
        pass

    def predict(self, features):
        preprocessing_path = os.path.join("artifacts/pickle/", "data_transformation.pkl")
        model_path = os.path.join("artifacts/pickle/", "model.pkl")
        preprocessing = load_pickle(preprocessing_path)
        model = load_pickle(model_path)
        processed_data = preprocessing.transform(features)
        op = model.predict(processed_data)
        return op


class GetFeature:
    def __init__(self, gender, age: int, smoking: str, yellow_fingers: str, anxiety: str, peer_pressure: str,
                 chronic_disease: str,
                 fatigue: str, allergy: str, wheezing: str, alcohol_consuming: str, coughing: str,
                 shortness_of_breath: str,
                 swallowing_difficulty: str, chest_pain: str) :
        self.age = age,
        self.gender = gender
        self.smoking = smoking
        self.yellow_fingers = yellow_fingers
        self.anxiety = anxiety
        self.peer_pressure = peer_pressure
        self.chronic_disease = chronic_disease
        self.fatigue = fatigue
        self.allergy = allergy
        self.wheezing = wheezing
        self.alcohol_consuming = alcohol_consuming
        self.coughing = coughing
        self.shortness_of_breath = shortness_of_breath
        self.swallowing_difficulty = swallowing_difficulty
        self.chest_pain = chest_pain

    def to_dataframe(self):
        feature = {
            "gender": [self.gender],
            "age": [self.age[0]],
            "smoking": [self.smoking],
            "yellow_fingers": [self.yellow_fingers],
            "anxiety": [self.anxiety],
            "peer_pressure": [self.peer_pressure],
            "chronic_disease": [self.chronic_disease],
            "fatigue": [self.fatigue],
            "allergy": [self.allergy],
            "wheezing": [self.wheezing],
            "alcohol_consuming": [self.alcohol_consuming],
            "coughing": [self.coughing],
            "shortness_of_breath": [self.shortness_of_breath],
            "swallowing_difficulty": [self.swallowing_difficulty],
            "chest_pain": [self.chest_pain]
        }

        feature = pd.DataFrame(feature)
        print(feature)
        return feature
