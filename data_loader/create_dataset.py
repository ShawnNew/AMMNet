
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
from preprocessor.Preprocessor import Preprocessor

class adobeDataset(Dataset):
    def __init__(self, root_, train=True, transform=None):
        self.root = os.path.expanduser(root_)
        self.transform = transform
        self.trainable = train # indicate whether generate train set or test set

        preprocessor = Preprocessor(self.root, shuffle=False)

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
        sample_dict = {}
        for item in items_list:
            sample_dict[item.split("/")[0]] = os.path.join(self.root, item)
                
