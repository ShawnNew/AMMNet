from torchvision import transforms
from base import BaseDataLoader
from .create_dataset import adobeDataset
from .preprocessor.utils import MultiRescale, RandomCrop, MultiToTensor



class AMSMNetDataLoader(BaseDataLoader):
    """
    adobe alpha matting datasets loading
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        
        if training:
            trsfm = transforms.Compose([
                RandomCrop(224),
                MultiRescale([1, 0.5, 0.25]),
                MultiToTensor()   # scale to [0, 1]
            ])
        else:
            trsfm = transforms.Compose([
                MultiRescale([1, 0.5, 0.25]),
                MultiToTensor()   # scale to [0, 1]
            ])
        self.data_dir = data_dir
        self.dataset = adobeDataset(self.data_dir, train=training, transform=trsfm, shuffle=False)
        super(AMSMNetDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        