import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer



@dataclass # it is a decoroter that have classes line __init__ and etc already installed
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv") # os.path.join will make a file or directory in the given path which in this case is artifacts folder
    test_data_path: str=os.path.join('artifacts',"test.csv")# same goes with test and data
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() # ingestion_config is a variable that consist of the above three values
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook/data/stud.csv') # we can choose any dataset that we need
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) #  This line creates the directory (or directories) in which the file train_data_path will be stored if they don't already exist.

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) # ye ingestion_config variable se raw data ka path lega and csv me convert kar dega

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) # ye ingestion_config varaibe se train data ka path lega aur usse csv me convert kar dega
               # header is alibrary in pandas which mean to include headern name in csv file
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)# same with test data

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path, # important for data_transformation
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__": #this line of code checks wether this program is running as a main program aur used as import
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))



