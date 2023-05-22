import os,sys 
import pandas as pd
from typing import Optional
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from music.config import k
from music import utils
from music.logger import logging
from music.exception import MusicException
from music.entity import artifact_entity,config_entity
from music.entity.artifact_entity import DataTransformationArtifact


class ModelTrainer:

    def __init__(self, model_trainer_config:config_entity.ModelTrainerConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact,
                data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
    
        except Exception as e:
            raise MusicException(e, sys)

    def train_model(self, features, k):
        """
        Model training
        """
        try:
            kmeans =  KMeans(n_clusters=3)
            kmeans.fit(features)
            return kmeans

        except Exception as e:
            raise MusicException(e, sys)

    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        """
        Preparing dataset
        """
        try:
            logging.info("Loading train and test array.")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)
            df = pd.read_csv(self.data_ingestion_artifact.feature_store_file_path)
            logging.info("Train the model")
            model = self.train_model(train_arr)

            print(model.labels_)

            df['predicted_label'] = model.labels_

            lebel_encoder = DataTransformationArtifact.target_encoder_path

            # Calcualting the accuracy score
            accuracy_score = accuracy_score(lebel_encoder.inverse_transform(df['label']), lebel_encoder.inverse_transform(df['predicted_label']))

            logging.info("Checking if our model is a good model or not")
            if accuracy_score<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {accuracy_score}")

            # Saving trained model if it passes using utils
            logging.info("Saving mode object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            # Prepare artifact
            logging.info("Prepare the artifact")
            model_trainer_artifact  = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path, 
            f1_train_score=f1_train_score, f1_test_score=f1_test_score)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
            
        except Exception as e:
            raise ThyroidException(e, sys)

