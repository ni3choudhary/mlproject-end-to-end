import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        

        try:

            numerical_columns = ['age', 'avg_glucose_level', 'bmi']

            categorical_columns = [
                'gender', 
                'ever_married', 
                'work_type',
                'Residence_type', 
                'smoking_status',
                'hypertension', 
                'heart_disease'
                ]

            numerical_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(missing_values=np.nan, strategy="mean")),
                ("scaler",StandardScaler())

                ]
            )
            logging.info(f"Numerical Columns: {numerical_columns}")

            categorical_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehot-encoder",OneHotEncoder(handle_unknown="ignore",drop='first'))
                ]

            )

            logging.info(f"Categorical Columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",numerical_pipeline,numerical_columns),
                ("cat_pipelines",categorical_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            logging.exception(CustomException(e,sys))
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            columns_to_convert = ['hypertension', 'heart_disease']
            train_df[columns_to_convert] = train_df[columns_to_convert].astype(str)
            test_df[columns_to_convert] = test_df[columns_to_convert].astype(str)
            logging.info("Converted Numerical Columns to Categorical Columns Successfully...")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            drop_feature = 'id'
            target_column_name="stroke"

            input_feature_train_df = train_df.drop(columns=[drop_feature,target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[drop_feature,target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.exception(CustomException(e,sys))
            raise CustomException(e,sys)