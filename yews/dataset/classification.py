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

