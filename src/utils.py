import pandas as pd
import os
import pickle
from src.exception import CustomException
from src.logger import logging
import sys

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.exception(CustomException(e,sys))
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model
            
            y_train_pred = model.predict(X_train)
            y_train_pred_proba = model.predict_proba(X_train)[:, 1]

            y_test_pred = model.predict(X_test)
            y_test_pred_proba = model.predict_proba(X_test)[:, 1]

            train_model_score = roc_auc_score(y_train, y_train_pred_proba)

            test_model_score = roc_auc_score(y_test, y_test_pred_proba)

            report[list(models.keys())[i]] = test_model_score

        logging.info("Model Fitted Successfully...")
        return report

    except Exception as e:
        logging.exception(CustomException(e,sys))
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.exception(CustomException(e,sys))
        raise CustomException(e, sys)