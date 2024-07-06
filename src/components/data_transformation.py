import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.logger import logging
from src.exception import CustomException
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            
            numerical_features = ["reading_score","writing_score"]
            categorical_features = ["gender", "race_ethnicity", "parental_level_of_education","lunch", "test_preparation_course"]

            num_pipeline = Pipeline (
                steps = [
                    ("imputer", SimpleImputer( strategy= "median" )),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer (strategy= "most_frequent")),
                    ("encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f" Numerical features are {numerical_features}")

            logging.info(f"Cat features are {categorical_features}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline,categorical_features)
                ]
            )


            logging.info("Num features standard scaling complete")

            logging.info("Cat features encoding complete")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    def initiate_data_transformation(self, train_path, test_path):
        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor_obj = self.get_data_transformer_object()

            target_column = "math_score"

            numerical_features = ["reading_score","writing_score"]
            categorical_features = ["gender", "race_ethnicity", "parental_level_of_education","lunch", "test_preparation_course"]

            input_train_df = train_df.drop(columns= [target_column],axis=1)
            target_train_df = train_df[target_column]
            input_test_df = test_df.drop(columns= [target_column],axis=1)
            target_test_df = test_df[target_column]

            logging.info(f"Applying preprocessing obj on training and test df")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_train_df)
            input_feature_test_arr = preprocessor_obj.fit_transform(input_test_df) 

            train_arr = np.c_[input_feature_train_arr, np.array(input_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(input_test_df)]       

            logging.info(f"Saved preprocessing obj")

            save_object(
                filepath = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj

            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)