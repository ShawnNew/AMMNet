import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
#from .preprocessor.Preprocessor import Preprocessor
from .preprocessor.utils import generate_gradient_map

class myDataset(Dataset):
    def __init__(self, root_, train, transform, shuffle):
        self.root = os.path.expanduser(root_)
        self.transform = transform
        self.trainable = train # indicate whether generate train set or test set
        
        train_lists_ = []
        test_lists_ = []

        train_lists_ += [os.path.join(self.root, x) \
                        for x in os.listdir(self.root) \
                        if os.path.isfile(os.path.join(self.root, x)) and\
                        x.endswith('txt') and 'train' in x]
        test_lists_ += [
                        os.path.join(self.root, x) \
                        for x in os.listdir(self.root) \
                        if os.path.isfile(os.path.join(self.root, x)) and\
                        x.endswith('txt') and 'test' in x]

        if self.trainable:
            data_file_ = []
            for path in train_lists_:
                with open(path, 'r') as f:
                    data_file_ += f.readlines()
            self.data_file_ = data_file_
            self.len_ = len(self.data_file_)
        else:
            data_file_ = []
            for path in test_lists_:
                with open(path, 'r') as f:
                    data_file_ += f.readlines()
            self.data_file_ = data_file_
            self.len_ = len(self.data_file_)
        if shuffle:
            random.shuffle(self.data_file_)

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        raise NotImplementedError


class adobeDataset(myDataset):
    def __init__(self, root_, train=True, transform=None, shuffle=False):
        super(adobeDataset, self).__init__(root_, train, transform, shuffle)

    def __getitem__(self, idx):
        # return the idx's image and related information
        line = self.data_file_[idx]
        items_list = line.rstrip().replace('./', '').split(' ')
        img_path = os.path.join(self.root, items_list[1])
        gt_path = os.path.join(self.root, items_list[3])
        trimap_path = os.path.join(self.root, items_list[6])
        gradient_path = os.path.join(self.root, items_list[0])
        img = Image.open(img_path)           # h*w*c
        gt = Image.open(gt_path)             # h*w
        trimap = Image.open(trimap_path)     # h*w
        gradient = Image.open(gradient_path) # h*w
        # gradient = generate_gradient_map(np.asarray(gradient), 3)
        # gradient = Image.fromarray(gradient)
        if img.mode != 'RGB': img = img.convert('RGB')

        sample = {
            'name': img_path,
            'image': img,
            'gt': gt,
            'trimap': trimap,
            'gradient': gradient
        }
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class alphamatting(myDataset):
    def __init__(self, root_, train=True, transform=None, shuffle=False):
        super(alphamatting, self).__init__(root_, train, transform, shuffle)

    def __getitem__(self, idx):
        line = self.data_file_[idx]
        items_list = line.rstrip().replace('./', '').split(' ')
        if self.trainable:
            # modify here for different datasets
            img_path = os.path.join(self.root, items_list[0])
            gt_path = os.path.join(self.root, items_list[2])
            trimap_path = os.path.join(self.root, items_list[1])
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            trimap = Image.open(trimap_path)

            sample = {
                'name': img_path,
                'image': img,
                'gt': gt,
                'trimap': trimap
            }  
        else:
            img_path = os.path.join(self.root, items_list[0])
            trimap_path = os.path.join(self.root, items_list[1])
            img = Image.open(img_path)
            trimap = Image.open(trimap_path)
            
            sample = {
                'name': img_path,
                'image': img,
                'trimap': trimap
            }
        
        if self.transform:
            sample = self.transform(sample)

        return sample

