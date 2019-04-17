from .base import PathDataset
import numpy as np

__all__ = [
    'DirDataset',
    'DatasetArrayFolder',
    'DatasetFolder',
]


class DirDataset(PathDataset):
    """An abstract class representing a Dataset in a directory.

    Args:
        root (object): Directory of the dataset.
        sample_transform (callable, optional): A function/transform that takes
            a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            a target and transform it.

    Attributes:
        samples (list): List of samples in the dataset.
        targets (list): List of targets in teh dataset.

    """

    def __init__(self, **kwargs):
        super(DirDataset, self).__init__(**kwargs)
        if not self.root.is_dir():
            raise ValueError(f"{self.root} is not a directory.")


class DatasetArrayFolder(DirDataset):
    """A generic data loader for a folder of ``.npy`` files where samples are
    arranged in the following way: ::

        root/samples.npy: each row is a sample
        root/targets.npy: each row is a label

    where both samples and targets can be arrays.

    Args:
        root (object): Path to the dataset.
        sample_transform (callable, optional): A function/transform that takes
            a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            a target and transform it.

    Attributes:
        samples (list): List of samples in the dataset.
        targets (list): List of targets in the dataset.

    """

    def build_dataset(self):
        samples = np.load(self.root / 'samples.npy', mmap_mode='r')
        targets = np.load(self.root / 'targets.npy', mmap_mode='r')

        return samples, targets


class DatasetFolder(DirDataset):
    """A generic data loader for a folder where samples are arranged in the
    following way: ::

        root/.../class_x.xxx
        root/.../class_x.sdf3
        root/.../class_x.asd932

        root/.../class_y.yyy
        root/.../class_y.as4h
        root/.../blass_y.jlk2

    Args:
        root (path): Path to the dataset.
        loader (callable): Function that load one sample from a file.
        sample_transform (callable, optional): A function/transform that takes
            a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            a target and transform it.

    Attributes:
        samples (list): List of samples in the dataset.
        targets (list): List of targets in teh dataset.


    """

    class FilesLoader(object):
        """A dataset-like class for loading a list of files given a loader.

        Args:
            files (list): List of file paths
            loader (callable): Function that load one file.
        """

        def __init__(self, files, loader):
            self.files = files
            self.loader = loader

        def __getitem__(self, index):
            return self.loader(self.files[index])

        def __len__(self):
            return len(self.files)

    def __init__(self, loader, **kwargs):
        self.loader = loader
        super(DatasetFolder, self).__init__(**kwargs)

    def build_dataset(self):
        files = [p for p in self.root.glob("**/*") if p.is_file()]
        labels = [p.name.split('.')[0] for p in files]
        samples = self.FilesLoader(files, self.loader)

        return samples, labels

