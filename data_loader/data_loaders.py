from torchvision import transforms
from base import BaseDataLoader
from .create_dataset import adobeDataset, alphamatting, carmediaDataset
from .preprocessor.utils import MultiRescale, RandomCrop, MultiToTensor
from . import human_dataset


class AMSMNetDataLoader(BaseDataLoader):
    """
    adobe alpha matting datasets loading
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        
        if training:
            trsfm = transforms.Compose([
                # RandomCrop(480),
                MultiRescale(320),
                MultiToTensor()   # scale to [0, 1]
            ])
        else:
            trsfm = transforms.Compose([
                MultiRescale(320),
                MultiToTensor()   # scale to [0, 1]
            ])
        self.data_dir = data_dir
        self.dataset = adobeDataset(self.data_dir, train=training, transform=trsfm, shuffle=False)
        super(AMSMNetDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class AMSMNetTestDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        if training:
            trsfm = transforms.Compose([
                # RandomCrop(320),
                MultiRescale(480),
                MultiToTensor()   # scale to [0, 1]
            ])
        else:
           trsfm = transforms.Compose([
                # RandomCrop(320),
                MultiRescale(480),
                MultiToTensor()   # scale to [0, 1]
            ])
        self.data_dir = data_dir
        self.dataset = alphamatting(self.data_dir, train=training, transform=trsfm, shuffle=False)
        super(AMSMNetTestDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class AMSMNetCarMediaDataLoader(BaseDataLoader):
    """
    adobe alpha matting datasets loading
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        
        if training:
            trsfm = transforms.Compose([
                # RandomCrop(320),
                MultiRescale(480),
                MultiToTensor()   # scale to [0, 1]
            ])
        else:
            trsfm = transforms.Compose([
                MultiRescale(480),
                MultiToTensor()   # scale to [0, 1]
            ])
        self.data_dir = data_dir
        self.dataset = carmediaDataset(self.data_dir, train=training, transform=trsfm, shuffle=False)
        super(AMSMNetCarMediaDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class HumanDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        self.data_dir = data_dir
        self.dataset = human_dataset.human_matting_data(self.data_dir, 320)
        super(HumanDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

