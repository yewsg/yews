from pathlib import Path

from torch.utils import data


def is_dataset(obj):
    """Verfy if a object is a ``dataset-like`` object.

    """
    return getattr(obj, '__getitem__', None) and getattr(obj, '__len__', None)

def _format_transform_repr(transform, head):
    lines = transform.__repr__().splitlines()
    return (["{}{}".format(head, lines[0])] +
            ["{}{}".format(" " * len(head), line) for line in lines[1:]])


class BaseDataset(data.Dataset):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``build_dataset`` which construct the dataset-like object from root.

    A dataset-like object has both ``__len__`` and ``__getitem__`` implmented.
    Typical dataset-like objects include python list and numpy ndarray.

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
        """Method to construct ``samples`` and ``targets`` from ``self.root``.

        Returns:
            samples (ndarray): List of samples.
            labels (ndarray): List of labels.

        """
        raise NotImplementedError

    def __getitem__(self, index):
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
            body += _format_transform_repr(self.sample_transform,
                                           "Sample transforms: ")
        if self.target_transform is not None:
            body += _format_transform_repr(self.target_transform,
                                           "Target transforms: ")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

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
