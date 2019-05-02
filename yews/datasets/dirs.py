import numpy as np

from . import utils
from .base import PathDataset

__all__ = [
    'DirDataset',
    'DatasetArrayFolder',
    'DatasetFolder',
]

def get_files_under_dir(path, pattern):
    return [p for p in path.glob(pattern) if p.is_file()]


class DirDataset(PathDataset):
    """An abstract class representing a Dataset in a directory.

    Args:
        path (object): Path to the directory.
        sample_transform (callable, optional): A function/transform that takes
            a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            a target and transform it.

    Attributes:
        samples (list): List of samples in the dataset.
        targets (list): List of targets in teh dataset.

    """

    def is_valid(self):
        """Determine if root is a valid directory.

        """
        path_exists = super().is_valid()
        valid_dir = self.root.is_dir()
        return path_exists and valid_dir

    def handle_invalid(self):
        raise ValueError(f"{self.root} is not a valid directory.")


class DatasetArrayFolder(DirDataset):
    """A generic data loader for a folder of ``.npy`` files where samples are
    arranged in the following way: ::

        root/samples.npy: each row is a sample
        root/targets.npy: each row is a target

    where both samples and targets can be arrays.

    Args:
        path (object): Path to the dataset folder.
        sample_transform (callable, optional): A function/transform that takes
            a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            a target and transform it.

    Attributes:
        samples (list): List of samples in the dataset.
        targets (list): List of targets in the dataset.

    """

    def is_valid(self):
        """Determine if root is a valid array folder.

        """
        valid_dir = super().is_valid()
        has_samples = (self.root / 'samples.npy').exists()
        has_targets = (self.root / 'targets.npy').exists()
        return  valid_dir and has_samples and has_targets

    def handle_invalid(self):
        raise ValueError(f"{self.root} is not a valid array folder. Requires samples.npy and targets.npy.")

    def build_dataset(self):
        """Returns samples and targets.

        """
        samples_path = self.root / 'samples.npy'
        targets_path = self.root / 'targets.npy'

        samples = utils.load_npy(samples_path)
        targets = utils.load_npy(targets_path)

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
        path (path): Path to the dataset.
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
        # TO-DO: check if root directory is empty
        self.loader = loader
        super().__init__(**kwargs)

    def build_dataset(self):
        """Return samples and targets.

        """
        files = get_files_under_dir(self.root, '**/*')
        labels = [p.name.split('.')[0] for p in files]
        samples = self.FilesLoader(files, self.loader)

        return samples, labels
