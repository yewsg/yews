from pathlib import Path

from torch.utils import data

from .utils import create_npy
from .utils import load_npy
from .utils import tqdm

def is_dataset(obj):
    r"""Verfy if a object is ``dataset-like`` defined in
    :class:`torch.utils.data.Dataset`.

    Args:
        obj: Object to be determined.

    Returns:
        bool: True for ``dataset-like`` object, false otherwise.

    """
    return hasattr(obj, '__getitem__') and hasattr(obj, '__len__')


class BaseDataset(data.Dataset):
    r"""An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``build_dataset`` which construct the dataset-like object from root.

    Note:
        A dataset-like object has both ``__len__`` and ``__getitem__``
        impolemented.

    Args:
        root (object): Source of the dataset.
        sample_transform (callable, optional): A function/transform that takes
            a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            a target and transform it.

    Attributes:
        samples (dataset-like object): Dataset-like object for samples.
        targets (dataset-like object): Dataset-like object for targets.


    """

    _repr_indent = 4

    def __init__(self, root=None, sample_transform=None, target_transform=None):
        self.root = root

        if self.is_valid():
            self.samples, self.targets = self.build_dataset()
            # verify self.samples and self.targets are dataset-like object
            if not is_dataset(self.samples):
                raise ValueError("self.samples is not a dataset-like object")
            if not is_dataset(self.targets):
                raise ValueError("self.targets is not a dataset-like object")

            if len(self.samples) == len(self.targets):
                self.size = len(self.targets)
            else:
                raise ValueError("Samples and targets have different lengths.")
        else:
            self.handle_invalid()

        self.sample_transform = sample_transform
        self.target_transform = target_transform

    def is_valid(self):
        return self.root is not None

    def handle_invalid(self):
        self.size = 0

    def build_dataset(self):
        """Construct ``samples`` and ``targets`` from ``self.root``.

        Returns:
            Constructed dataset-like objects of samples and targets. They will
            be stored in ``self.samples`` and ``self.targets`` by ``__init__``.

        """
        raise NotImplementedError

    def export_dataset(self, path):
        """Export ``self.samples`` and ``self.targets`` to ``npy`` files.

        Returns:
            create two ``npy`` files at the given ``path``.
        """
        path = Path(path)
        # get array shape
        samples_shape = (self.__len__(), ) + self.__getitem__(0)[0].shape
        targets_shape = (self.__len__(), ) + self.__getitem__(0)[1].shape
        # get array dtype
        samples_dtype = self.__getitem__(0)[0].dtype
        targets_dtype = self.__getitem__(0)[1].dtype
        # reserve disk space
        fs = create_npy(path / 'samples.npy', samples_shape, samples_dtype)
        ft = create_npy(path / 'targets.npy', targets_shape, targets_dtype)

        # populate memmap numpy array
        for i in range(self.__len__()):
            # add one item in the dataset
            data_point = self[i]
            fs[i] = data_point[0]
            ft[i] = data_point[1]

            if (i % 100):
                print(f"Exporting {i} / {self.__len__()}")

        del fs
        del ft

    def __getitem__(self, index):
        """Indexing the dataset.

        Args:
            index (int): Index.

        Returns:
            Tuple of (samples, targets) combination.

        """
        sample = self.samples[index]
        target = self.targets[index]

        if self.sample_transform is not None:
            sample = self.sample_transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return self.size

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if self.sample_transform is not None:
            body += self._format_transform_repr(self.sample_transform,
                                                "Sample transforms: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transforms: ")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    @staticmethod
    def _format_transform_repr(transform, head):
        """Format transform representation for ``BaseDataset``.

        Args:
            transform (transform-like): Transform to be formated
            head (str): Formating string.

        Returns:
            str: Formated string.

        """
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""


class PathDataset(BaseDataset):
    """An abstract class representing a Dataset defined by a Path.

    Args:
        path (object): Path to the dataset.
        sample_transform (callable, optional): A function/transform that takes
            a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            a target and transform it.

    Attributes:
        samples (list): List of samples in the dataset.
        targets (list): List of targets in teh dataset.

    """

    def __init__(self, path, **kwargs):
        path = Path(path).resolve()    # make a Path object regardless of existence
        super().__init__(root=path, **kwargs)

    def is_valid(self):
        """Determine if the root path is valid.

        Other subclasses should overload this method if valid paths are defined
        differently.

        """

        return self.root.exists()

    def handle_invalid(self):
        raise ValueError(f"{self.root} is not a valid path.")


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


class FileDataset(PathDataset):
    """An abstract class representing a Dataset in a file.

    Args:
        path (object): File of the file.
        sample_transform (callable, optional): A function/transform that takes
            a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            a target and transform it.

    Attributes:
        samples (list): List of samples in the dataset.
        targets (list): List of targets in teh dataset.

    """

    def is_valid(self):
        return self.root.is_file()

    def handle_invalid(self):
        raise ValueError(f"{self.root} is not a file.")


class Dataset(DirDataset):
    """Yew's standard data loader for a folder of ``.npy`` files where samples
    are arranged in the following way: ::

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

        samples = load_npy(samples_path)
        targets = load_npy(targets_path)

        return samples, targets
