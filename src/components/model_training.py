from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from src.utils import save_file_as_pickle
import os
from dataclasses import dataclass

@dataclass
class ModelTrainingConfig:
    trained_model_path=os.path.join("artifacts/pickle","model.pkl")

class ModelTraining:
    def __init__(self):
        self.trained_model=ModelTrainingConfig()
    def initiate_model_training(self,transformed_train_dataset,transformed_test_dataset):
        try:
            xtrain=transformed_train_dataset[:,:-1]
            ytrain=transformed_train_dataset[:,-1]
            xtest=transformed_test_dataset[:,:-1]
            ytest=transformed_test_dataset[:,-1]

            models={
                "logisticsRegression":LogisticRegression(),
                "SupportVectorMachine":SVC(),
                "DecisionTree":DecisionTreeClassifier(),
                "Randomforest":RandomForestClassifier(),
                "adaboost":AdaBoostClassifier(),
                "gradientboost":GradientBoostingClassifier()
            }
            model_report={}
            for model_name,model in models.items():
                model.fit(xtrain,ytrain)
                predicted_value=model.predict(xtest)
                accuracy=accuracy_score(ytest,predicted_value)
                model_report[model_name]=accuracy

            best_model_name=max(model_report,key=model_report.get)
            best_model_accuracy=max(model_report.values())

            best_model=models[best_model_name]
            print(best_model_name,best_model_accuracy)

            save_file_as_pickle(best_model,self.trained_model.trained_model_path)

        except Exception as e:
            raise e