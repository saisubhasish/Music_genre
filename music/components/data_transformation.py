import os,sys 
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from typing import Optional

from music import utils
from music.entity import artifact_entity,config_entity
from music.exception import MusicException
from music.logger import logging
from music.config import TARGET_COLUMN



class DataTransformation:

    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                    data_validation_artifact:artifact_entity.DataValidationArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config=data_transformation_config
            self.data_validation_artifact=data_validation_artifact
        except Exception as e:
            raise MusicException(e, sys)

        
    @classmethod
    def get_data_transformer_object(cls)->Pipeline:     # Attributes of this class will be same across all the object 
        try:
            robust_scaler =  StandardScaler()
            pipeline = Pipeline(steps=[
                    ('RobustScaler',robust_scaler)  # To handle outliers in one side of distribution
                ])
            return pipeline
        except Exception as e:
            raise MusicException(e, sys)
        
    @classmethod
    def get_label_encoder_object(cls)->Pipeline:     # Attributes of this class will be same across all the object 
        try:
            label_encoder =  LabelEncoder()
            pipeline = Pipeline(steps=[
                    ('encoder',label_encoder)  # To handle outliers in one side of distribution
                ])
            return pipeline
        except Exception as e:
            raise MusicException(e, sys)
    

    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            # Reading training and testing file
            logging.info("Reading training and testing file")
            train_df = pd.read_csv(self.data_validation_artifact.train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.test_file_path)          
            
            # Selecting input feature for train and test dataframe
            logging.info("Selecting input feature for train and test dataframe")
            input_feature_train_df=train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df=test_df.drop(TARGET_COLUMN,axis=1)

            # Selecting target feature for train and test dataframe
            logging.info("Selecting target feature for train and test dataframe")
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Encoding the target feature values
            logging.info("Encoding the target feature values")
            label_encoder = self.get_label_encoder_object()
            label_encoder.fit(target_feature_train_df)

            # Transformation on target columns
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)   
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)
            logging.info(f"Target feature label encoded values: {target_feature_test_arr}")

            # Standardizing features
            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit_transform(input_feature_train_df)
            logging.info(input_feature_train_df.columns)
            
            # Imputing null values
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)  
            features_names = list(transformation_pipeline.feature_names_in_)                      #####    To handle the features in test set
            input_feature_test_df = input_feature_test_df[features_names]
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)        ##### 

            # Target encoder
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]    # concatenated transpose array
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            # Save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path, array=train_arr)
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path, array=test_arr)

            # Saving object
            utils.save_object(file_path=self.data_transformation_config.knn_imputer_object_path, obj=transformation_pipeline)
            utils.save_object(file_path=self.data_transformation_config.target_encoder_path, obj=label_encoder)

            # Preparing Artifact
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                knn_imputer_object_path=self.data_transformation_config.knn_imputer_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path)

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
            
        except Exception as e:
            raise MusicException(e, sys)
