import os
# import h5py
import random
import cv2
import numpy as np
from PIL import Image
import sys
import pdb

class Preprocessor:
    # class member
    random_crop_list_ = [320, 480, 640]
    flip_list_ = [True, False]
    # class constructor
    def __init__(self, root, shuffle=False):
        self.root_dir_ = root
        self.shuffle = shuffle

        _dict = {}
        for item in os.listdir(self.root_dir_):
            _path = os.path.join(self.root_dir_, item)
            if (os.path.isdir(_path)):  
                _dict[item] = getFileList(self.root_dir_, item)  # relative path against root
        
        self.dataset_dict = _dict
        small_count = sys.maxsize
        small_folder = " "
        for key, value in self.dataset_dict.items():
            len_ = len(value)
            if (small_count > len_):
                small_count = min(small_count, len_)
                small_folder = key
            else:
                pass

        total_num = small_count
        file_list = self.dataset_dict[small_folder]
        self.len_train = total_num - 100
        self.len_test = 100

        if self.shuffle:
            random.shuffle(file_list)
            self.splitTrainTest(file_list)

    def splitTrainTest(self, file_list):
        """
        Parse the dataset directory and record the relative path of images 
        within each folder into a dictionary.
        """
        split_list = [file_list[:self.len_train], file_list[self.len_train:]]
        train_file = os.path.join(self.root_dir_, 'train.txt')
        test_file = os.path.join(self.root_dir_, 'test.txt')
        writeDataSetFile(self.dataset_dict, [train_file, test_file], split_list)

    def getTrainFile(self):
        return os.path.join(self.root_dir_, 'train.txt')

    def getTestFile(self):
        return os.path.join(self.root_dir_, 'test.txt')


def getFileList(base, sub):
    """
    Get file list of a directory:
    Param:
        base: base directory
        sub: sub-directory name
    Return:
        a list of image file name
    """
    path = os.path.join(base, sub)
    files = os.listdir(path)
    fileList = []
    for f in files:
        if (os.path.isfile(path + '/' + f)):
            path_ = './' + sub
            path_ = os.path.join(path_, f)
            # add image file into list
            fileList.append(path_)
    return fileList

def writeDataSetFile(dict_, fns, list_):
    for i in range(len(fns)):
        with open(fns[i], 'w') as f:
            for item in list_[i]:
                # f.write(dict_.values)
                for key in dict_.keys():
                    temp_path = './' + \
                                key + \
                                '/' + \
                                os.path.basename(item)
                    f.write(temp_path + ' ')
                f.write('\n')

