from pathlib import Path
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):
    """Basic dataset class for classification task.

    """

    _repr_indent = 4

    def __init__(self, source=None, transform=None, target_transform=None):
        self.source = source
        self.transform = transform
        self.target_transform = target_transform

        self.classes = self._find_classes()
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = self._make_dataset()
        self.targets = [self.class_to_idx[s[1]] for s in self.samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self):
        raise NotImplementedError

    def _make_dataset(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.targets)

    def __repr__(self):
        head = f"Dataset {self.__class__.__name__}"
        body = [f"Number of datapoints: {self.__len__()}"]
        if type(self.source) is str:
            body.append(f"Source location: {self.source}")
        else:
            body.append(f"Source location: array")
        if hasattr(self, 'transform') and self.transform is not None:
            body.append(self._foramt_transform_repr(self.transform, "Transforms: "))
        if hasattr(self, 'targe_transform') and self.target_transform is not None:
            body.append(self._foramt_transform_repr(self.target_transform, "Target Transforms: "))

        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _foramt_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])


class DatasetArray(ClassificationDataset):
    """A generic dataloader for classification task where samples and targets
    are stored in numpy arrays.

    Args:
        samples (ndarray): Numpy array of seismic data where each row is a
            sample. The first column is a single/multi-component waveform. The
            second column is the target. The rest columns are additional info
            about the dataset.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

    Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample, class_index) tuples.
        targets (list): The class_index value for each sample in the dataset.
    """

    def _find_classes(self):
        classes = list(set(self.source[:, 1]))
        return classes

    def _make_dataset(self):
        return [(s[0], s[1]) for s in self.source]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = self.samples[index][0]
        target = self.targets[index]

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class DatasetFolder(ClassificationDataset):
    """A generic dataloader for classification task where the samples are
    arranaged in a folder:

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, transform=None, target_transform=None):
        super(ClassificationDatasetFolder, self).__init__(root, transform=transform,
                                                          target_transform=target_transform)
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


