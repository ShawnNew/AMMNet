from torchvision import datasets, transforms
from base import BaseDataLoader



class AMSMNetDataLoader(BaseDataLoader):
    """
    adobe alpha matting datasets loading
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(AMSMNetDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        