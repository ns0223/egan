import os
from PIL import Image
from torch.utils.data import Dataset


class SISRDataset(Dataset):
    """The data set loading class only needs to prepare high-resolution images.
    Args:
        root         (str): Training data set address.
    """

    def __init__(self, root):
        super(SISRDataset, self).__init__()
        self.filenames = [os.path.join(root, x) for x in os.listdir(root)]

    def __getitem__(self, index):
        hr = Image.open(self.filenames[index])
        return hr

    def __len__(self) -> int:
        return len(self.filenames)