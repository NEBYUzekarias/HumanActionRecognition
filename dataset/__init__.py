from ml_py.image import ImagePreprocessor
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import shutil

class HumanActionDataset(object):
    
    def __init__(self,dataset_path,image_shape,uses_generator=False):
        """A class that abstracts Human action dataset(www.nada.kth.se/cvap/actions/)
        
        Arguments:
            dataset_path {str} -- Path to folder containining folders containing images extracted from videos
            image_shape {tuple} -- Network input shape
        """

        self.dataset_path = dataset_path
        self.image_shape = image_shape
        self.uses_generator=uses_generator
        self.image_preprocessor = ImagePreprocessor()
    def __get_all_sequence_name(self):
        sequences = os.listdir(self.dataset_path)
        return sequences
    def split_train_test(self,validation = True):
       
        sequences = self.__get_all_sequence_name()
        
        train,test = train_test_split(sequences,test_size=0.2)
        train_df = pd.DataFrame(columns=["sequence"])
        train_df["sequence"] = train

        train_df.to_pickle(os.path.join(self.dataset_path,"train.pkl"))
        
        if validation:
            t,v = train_test_split(test,test_size=0.5)
            test_df = pd.DataFrame(columns=["sequence"])
            test_df["sequence"] = t
            validation_df = pd.DataFrame(columns=["sequence"])
            validation_df["sequence"] = v
            test_df.to_pickle(os.path.join(self.dataset_path,"test.pkl"))
            validation_df.to_pickle(os.path.join(self.dataset_path,"validation.pkl"))

        else:
            test_df = pd.DataFrame(columns=["sequence"])
            test_df["sequence"] = test
            test_df.to_pickle(os.path.join(self.dataset_path,"test.pkl"))
    
    def load_dataset(self,uses_generator=False):
        pass
    def generate_indexes(self,length):
        assert type(length)==int,"length should be integer value"
        assert length>0, "length value should be greater than 0"
        indexes = np.arange(length)
        np.random.shuffle(indexes)
        return indexes
    def generator(self,batch_size):
        if not self.uses_generator:
            raise Exception("Cannot use generator method when uses_generator attribute is false")

