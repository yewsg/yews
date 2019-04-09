from .classification import ClassificationDataset

class ClassificationDatasetArray(ClassificationDataset):
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
        sample = self.samples[index][0]
        target = self.targets[index]

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target



