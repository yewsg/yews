import os
from pathlib import Path

import numpy as np
import pytest

import yews.datasets as datasets
import yews.transforms as transforms

root_dir = Path('tests/assets').resolve()

@pytest.mark.smoke
def test_is_dataset():
    assert not datasets.is_dataset(0)
    assert datasets.is_dataset([])

def rm(path):
    try:
        os.remove(path)
    except OSError:
        pass

class DummpyDatasetlike(object):

    def __init__(self, size=1):
        self.size = size
        self.data = ['a item'] * self.size

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.size

@pytest.mark.smoke
class TestMandatoryMethods:

    def test_call_method(self):
        assert all([hasattr(getattr(datasets, t), '__getitem__') for t in
                    datasets.__all__])

    def test_repr_method(self):
        assert all([hasattr(getattr(datasets, t), '__len__') for t in
                    datasets.__all__])

@pytest.mark.smoke
class TestBaseDataset:

    class DummyBaseDataset(datasets.BaseDataset):

        def build_dataset(self):
            return DummpyDatasetlike(), DummpyDatasetlike()


    class DummyBaseDatasetNoSamples(datasets.BaseDataset):

        def build_dataset(self):
            return 0, DummpyDatasetlike()


    class DummyBaseDatasetNoTargets(datasets.BaseDataset):

        def build_dataset(self):
            return DummpyDatasetlike(), 0


    class DummyBaseDatasetWrongLength(datasets.BaseDataset):

        def build_dataset(self):
            return DummpyDatasetlike(1), DummpyDatasetlike(2)

    class DummyTransform(transforms.BaseTransform):

        def __call__(self, data):
            return "transformed"

    def test_empty_construct(self):
        dset = datasets.BaseDataset()
        assert len(dset) == 0

    def test_noempty_constrct(self):
        dset = self.DummyBaseDataset(root='.')
        assert len(dset) == 1

    def test_raise_notimplmenetederror(self):
        with pytest.raises(NotImplementedError):
            dset = datasets.BaseDataset('.')

    def test_no_samples(self):
        with pytest.raises(ValueError):
            dset = self.DummyBaseDatasetNoSamples(root='.')

    def test_no_targets(self):
        with pytest.raises(ValueError):
            dset = self.DummyBaseDatasetNoTargets(root='.')

    def test_samples_targets_not_match(self):
        with pytest.raises(ValueError):
            dset = self.DummyBaseDatasetWrongLength(root='.')

    def test_getitem_with_transform(self):
        dset = self.DummyBaseDataset(root='.',
                                     sample_transform=self.DummyTransform(),
                                     target_transform=self.DummyTransform())
        assert dset[0] == ('transformed', 'transformed')
        dset = self.DummyBaseDataset(root='.')
        assert dset[0] == ('a item', 'a item')

    def test_repr(self):
        dset = self.DummyBaseDataset(root='.',
                                     sample_transform='t',
                                     target_transform='tt')
        assert type(dset.__repr__()) is str
        dset = self.DummyBaseDataset()
        assert type(dset.__repr__()) is str


@pytest.mark.smoke
class TestPathDataset:

    class DummyPathDataset(datasets.PathDataset):

        def build_dataset(self):
            return DummpyDatasetlike(), DummpyDatasetlike()

    def test_root_is_path(self):
        # check existing path
        dset = self.DummyPathDataset(path='.')
        # check path resolved
        assert dset.root == Path(dset.root).resolve()
        # check non-existing path
        with pytest.raises(ValueError):
            dset = self.DummyPathDataset(path='abc')


class TestDirDataset:

    class DummyDirDataset(datasets.DirDataset):

        def build_dataset(self):
            return DummpyDatasetlike(), DummpyDatasetlike()

    def test_invalid_root(self):
        dset = self.DummyDirDataset(path='.')
        with pytest.raises(ValueError):
            dset = self.DummyDirDataset(path='setup.py')


class TestDatasetArrayFolder:

    def test_loading_npy(self):
        dset = datasets.DatasetArrayFolder(path=root_dir / 'array_folder')
        assert all([dset[0][0].shape == (3, 100), dset[0][1].shape == ()])

    def test_invalid_root(self):
        with pytest.raises(ValueError):
            dset = datasets.DatasetArrayFolder(path=root_dir)


class TestDatasetFolder:

    def test_loading_folder(self):
        dset = datasets.DatasetFolder(path=root_dir/ 'folder', loader=np.load)
        assert all([dset[0][0].shape == (3, 100), type(dset[0][1]) is str])


class TestFileDataset:

    class DummpyFileDataset(datasets.FileDataset):

        def build_dataset(self):
            return DummpyDatasetlike(), DummpyDatasetlike()

    def test_file_check(self):
        dset = self.DummpyFileDataset(path='setup.py')
        with pytest.raises(ValueError):
            dset = self.DummpyFileDataset(path='.')


class TestDatasetArray:

    def test_loading_array(self):
        dset = datasets.DatasetArray(path=root_dir / 'array/data.npy')
        assert all([dset[0][0].shape == (3, 100), dset[0][1].shape == ()])


class TestWenchuan:

    def test_download_flag(self):
        with pytest.raises(ValueError):
            datasets.Wenchuan(path='.', download=None)
        with pytest.raises(ValueError):
            datasets.Wenchuan(path='.')

    @pytest.mark.internet
    def test_wenchuan_download(self):
        # prepare root folder
        rm('samples.npy')
        rm('targets.npy')
        rm('wenchuan.tar.bz2')
        # test download and extract
        datasets.Wenchuan(path='.', download=True)
        # test extract only
        rm('samples.npy')
        rm('targets.npy')
        datasets.Wenchuan(path='.', download=True)
        # test ready dataset
        dset = datasets.Wenchuan(path='.', download=True)
        assert dset[0][1] == 1
        assert len(dset) == 60276
        # clean root folder
        rm('samples.npy')
        rm('targets.npy')
        rm('wenchuan.tar.bz2')
