import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
from .preprocessor.Preprocessor import Preprocessor
from .preprocessor.utils import generate_gradient_map

class adobeDataset(Dataset):
    def __init__(self, root_, train=True, transform=None, shuffle=False):
        self.root = os.path.expanduser(root_)
        self.transform = transform
        self.trainable = train # indicate whether generate train set or test set

        preprocessor = Preprocessor(self.root, 0, shuffle=shuffle)
        
        if self.trainable:
            with open(preprocessor.getTrainFile(), 'r') as f:
                self.data_file_ = f.readlines()
            self.len_ = preprocessor.len_train
        else:
            with open(preprocessor.getTestFile(), 'r') as f:
                self.data_file_ = f.readlines()
            self.len_ = preprocessor.len_test

    def __len__(self):
        # return the length of the dataset
        return self.len_

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

class alphamatting(Dataset):
    def __init__(self, root_, train=True, transform=None, shuffle=False):
        self.root = os.path.expanduser(root_)
        self.transform = transform
        self.trainable = train # indicate whether generate train set or test set
        if self.trainable:
            path_ = os.path.join(self.root, 'train.txt')
            with open(path_, 'r') as f:
                self.data_file_ = f.readlines()
                self.len_ = len(self.data_file_)
        else:
            path_ = [
                os.path.join(self.root, 'test-trimap1.txt'),
                os.path.join(self.root, 'test-trimap2.txt'),
                os.path.join(self.root, 'test-trimap3.txt')
            ]
            data_file_ = []
            for path in path_:
                with open(path, 'r') as f:
                    data_file_ += f.readlines()
            
            self.data_file_ = data_file_
            self.len_ = len(self.data_file_)

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        line = self.data_file_[idx]
        items_list = line.rstrip().replace('./', '').split(' ')
        if self.trainable:
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

