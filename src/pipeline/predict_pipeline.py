import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')

            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            preds_proba = model.predict_proba(data_scaled)[:, 1]
            return preds, preds_proba
        
        except Exception as e:
            logging.exception(CustomException(e,sys))
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        gender: str,
        ever_married: str,
        work_type: str,
        Residence_type: str,
        smoking_status: str,
        hypertension: str,
        heart_disease: str,
        age: int,
        avg_glucose_level: int,
        bmi: int 
        ):

        self.gender = gender
        self.ever_married = ever_married
        self.work_type = work_type
        self.Residence_type = Residence_type
        self.smoking_status = smoking_status
        self.hypertension = hypertension
        self.heart_disease = heart_disease
        self.age = age
        self.avg_glucose_level = avg_glucose_level
        self.bmi = bmi

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "ever_married": [self.ever_married],
                "work_type": [self.work_type],
                "Residence_type": [self.Residence_type],
                "smoking_status": [self.smoking_status],
                "hypertension": [self.hypertension],
                "heart_disease": [self.heart_disease],
                "age": [self.age],
                "avg_glucose_level": [self.avg_glucose_level],
                "bmi": [self.bmi],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            logging.exception(CustomException(e,sys))
            raise CustomException(e, sys)