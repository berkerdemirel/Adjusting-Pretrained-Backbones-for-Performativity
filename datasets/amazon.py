import torch
import numpy as np

from wilds.datasets.wilds_dataset import WILDSSubset
from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
from wilds.datasets.amazon_dataset import AmazonDataset


class DomainAmazon(AmazonDataset):

    def __init__(self, root_dir, download=True, transform=None, **kwargs):

        super(DomainAmazon, self).__init__(root_dir=root_dir, download=download, **kwargs)

        self.targets = self._y_array
        self.transform = transform

    def __getitem__(self, idx):
        x = self.get_input(idx)
        y = self.y_array[idx]
        # metadata = self.metadata_array[idx]

        if self.transform is not None:
           x = self.transform(x)
        return x, y

    @staticmethod
    def domains():
        return [
            "none"
        ]