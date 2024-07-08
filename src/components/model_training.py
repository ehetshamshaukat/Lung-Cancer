from sklearn.linear_model import LogisticRegression
from sklearn.svm import  SVC
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from src.utils import save_file_as_pickle
from sklearn.metrics import accuracy_score
from dataclasses import dataclass
import os

@dataclass
class ModelTrainingConfig:
    trained_model_pickle_path=os.path.join("artifacts/pickle","model.pkl")

class ModelTraining:
    def __init__(self):
        self.model_pickle=ModelTrainingConfig()

    def initiate_model_training(self,transformed_train_data,transformed_test_data):
        try:
            xtrain=transformed_train_data[:,:-1]
            ytrain=transformed_train_data[:,-1]

            xtest=transformed_test_data[:,:-1]
            ytest=transformed_test_data[:,-1]

            models={
                "logisticsRegression":LogisticRegression(),
                "svm":SVC(),
                "DecisionTree":DecisionTreeClassifier(),
                "Randomforest":RandomForestClassifier(),
                "Adaboost":AdaBoostClassifier(),
                "Gradientboost":GradientBoostingClassifier()

            }
            model_report={}
            for model_name,model in models.items():
                model.fit(xtrain,ytrain)
                predicted_value=model.predict(xtest)
                accuracy=accuracy_score(ytest,predicted_value)
                model_report[model_name]=accuracy

            best_model_name=max(model_report,key=model_report.get)
            best_model_accuracy=max(model_report.values())

            print("best model:",best_model_name)
            print("accuracy score:",best_model_accuracy * 100)

            best_model=models[best_model_name]
            save_file_as_pickle(best_model,self.model_pickle.trained_model_pickle_path)
        except Exception as e:
            raise e
