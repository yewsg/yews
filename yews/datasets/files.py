from pathlib import Path
import numpy as np

from .base import PathDataset

__all__ = [
    'FileDataset',
    'DatasetArray',
]


class FileDataset(PathDataset):
    """An abstract class representing a Dataset in a file.

    Args:
        root (object): File of the dataset.
        sample_transform (callable, optional): A function/transform that takes
            a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            a target and transform it.

    Attributes:
        samples (list): List of samples in the dataset.
        targets (list): List of targets in teh dataset.

    """

    def __init__(self, **kwargs):
        super(FileDataset, self).__init__(**kwargs)
        if not self.root.is_file():
            raise ValueError(f"{self.root} is not a file.")


class DatasetArray(FileDataset):
    """A generic data loader for ``.npy`` file where samples are arranged is
    the following way: ::

        array = [
            [sample0, target0],
            [sample1, target1],
            ...
        ]

    where both samples and targets can be arrays.

    Args:
        root (object): Path to the dataset.
        sample_transform (callable, optional): A function/transform that takes
            a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            a target and transform it.

    Attributes:
        samples (list): List of samples in the dataset.
        targets (list): List of targets in teh dataset.

    """

    def build_dataset(self):
        data = np.load(self.root)
        return data[:, 0], data[:, 1]

