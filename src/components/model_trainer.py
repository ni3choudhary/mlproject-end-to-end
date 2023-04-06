import os
import sys
import json

from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from src.utils import save_object, evaluate_models
from sklearn.metrics import roc_auc_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split into Training and Test Data...")

            X_train, X_test, y_train, y_test=(
                                        train_array[:,:-1],
                                        test_array[:,:-1],
                                        train_array[:,-1],
                                        test_array[:,-1]
                                    )

            # Models
            models = {
                "KNN": KNeighborsClassifier(n_neighbors = 3),
                "SVM": SVC(probability=True),
                "Decision Tree": DecisionTreeClassifier(random_state=2023),
                "Random Forest": RandomForestClassifier(random_state=2023),
                "XGBoost": XGBClassifier(random_state=2023),
                "CatBoost": CatBoostClassifier(random_state=2023,verbose=0)
            }
            
            with open("src/components/model_parameters.json", "r") as jsonfile:
                params = json.load(jsonfile)

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                logging.exception("No Best Model Found...")

            logging.info(f"Best Model {best_model_name} Found on both Training and Testing Dataset...")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            predicted_proba = best_model.predict_proba(X_test)[:, 1]

            roc_auc = roc_auc_score(y_test, predicted_proba)
            return roc_auc

        except Exception as e:
            logging.exception(CustomException(e,sys))
            raise CustomException(e,sys)