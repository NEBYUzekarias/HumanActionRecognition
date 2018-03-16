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
    def split_array(self,array,max_size):
        """Split given array to smaller array with maximum array length of max_size.
        If len(array)%max_size ==0 all arrays will have the same length(max_size), other 
        wise all arrays will have the same length(max_size) except the last one. It will contain
        len(array)%max_size.
        Arguments:
            array {list} -- List that will be splitted.
            max_size {int} -- Maximum allowable output array length
        
        Returns:
            list -- list containing splitted list
        """

        assert max_size>0, "max_size should be greater than 0"
        if max_size>=len(array):
            return array
        output = []
        for i in range(0,len(array),max_size):
            output+=[array[i:i+max_size]]
        return output
    def __split_sequence_helper(self,sequence,max_sequence_length):
        current_path  = os.path.join(self.dataset_path,sequence)
        s_images = os.listdir(current_path)
        s_images.sort()
        splited_seq = self.split_array(s_images,max_sequence_length) 
        output={}
        for i in range(len(splited_seq)):
            output[sequence+"_"+str(i)] = []
            for image_file in splited_seq[i]:
                output[sequence+"_"+str(i)].append(image_file)
        return output
    def __save_splitted_sequences(self,origin_dir,sequences_dict,output_path):
        for sequence in sequences_dict:
            if not os.path.exists(os.path.join(output_path,sequence)):
                os.mkdir(os.path.join(output_path,sequence))
            for img_file in sequences_dict[sequence]:
                shutil.copy(os.path.join(origin_dir,img_file),os.path.join(output_path,sequence,img_file))
            
    def split_all_sequences(self,output_path):
        train_df = pd.read_pickle(os.path.join(self.dataset_path,"train_all.pkl"))
        test_df = pd.read_pickle(os.path.join(self.dataset_path,"test_all.pkl"))
        validation_df = pd.read_pickle(os.path.join(self.dataset_path,"validation_all.pkl"))
        train_small_sequences = {}

        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if not os.path.exists(os.path.join(output_path,"train")):
            os.mkdir(os.path.join(output_path,"train"))
            
        if not os.path.exists(os.path.join(output_path,"test")):
            os.mkdir(os.path.join(output_path,"test"))
        if not os.path.exists(os.path.join(output_path,"validation")):
            os.mkdir(os.path.join(output_path,"validation"))


        for sequence in train_df["sequence"]:
            smaller_sequences = self.__split_sequence_helper(sequence,30)
            self.__save_splitted_sequences(os.path.join(self.dataset_path,sequence),smaller_sequences,os.path.join(output_path,"train"))
        for sequence in test_df["sequence"]:
            smaller_sequences = self.__split_sequence_helper(sequence,30)
            self.__save_splitted_sequences(os.path.join(self.dataset_path,sequence),smaller_sequences,os.path.join(output_path,"test"))
        for sequence in validation_df["sequence"]:
            smaller_sequences = self.__split_sequence_helper(sequence,30)
            self.__save_splitted_sequences(os.path.join(self.dataset_path,sequence),smaller_sequences,os.path.join(output_path,"validation"))


    
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

