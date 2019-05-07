import os
from pathlib import Path

import obspy
import pytest

from yews.datasets import utils


class TestObspyIO():

    def TestStream2Array(self):
        st = obspy.read()
        assert utils.stream2array(st).shape == (3, 3000)

    def TestReadFrameObspy(self):
        path = 'tests/assets/sac/*.sac'
        assert utils.read_frame_obspy(path).shape == (3, 3000)


class TestMemoeryLimit():

    def test_default_memory_limit(self):
        assert utils.get_memory_limit() == 2 * 1024 ** 3

    def test_set_memory_limit(self):
        default_memory_limit = utils.get_memory_limit()
        utils.set_memory_limit(1)
        assert utils.get_memory_limit() == 1
        utils.set_memory_limit(default_memory_limit)


class TestLoadNpy():

    npy_file = 'tests/assets/array.npy'
    default_memory_limit = utils.get_memory_limit()

    def test_default_memory_limit(self):
        utils.load_npy(self.npy_file)
        assert utils.get_memory_limit() == self.default_memory_limit
        memory_limit = utils.get_memory_limit()
        utils.set_memory_limit(1)
        utils.load_npy(self.npy_file)
        utils.set_memory_limit(memory_limit)

    def test_custom_memory_limit(self):
        assert utils.get_memory_limit() == self.default_memory_limit
        utils.load_npy(self.npy_file, memory_limit=1)
        assert utils.get_memory_limit() == self.default_memory_limit


class TestTarUtils():

    def test_tar_extraction(self):
        utils.extract_tar('tests/assets/test.tar.bz2')
        with Path('test.txt').open('r') as f:
            assert f.readlines() == ['test\n']
        os.remove('test.txt')


class TestUrlUtils():

    def test_sizeof_fmt(self):
        assert utils.sizeof_fmt(10, 'm') == '10.0m'
        assert utils.sizeof_fmt(1e1) == '10.0B'
        assert utils.sizeof_fmt(1e4) == '9.8KB'
        assert utils.sizeof_fmt(1e7) == '9.5MB'
        assert utils.sizeof_fmt(1e10) == '9.3GB'
        assert utils.sizeof_fmt(1e13) == '9.1TB'
        assert utils.sizeof_fmt(1e16) == '8.9PB'
        assert utils.sizeof_fmt(1e19) == '8.7EB'
        assert utils.sizeof_fmt(1e22) == '8.5ZB'
        assert utils.sizeof_fmt(1e25) == '8.3YB'

    @pytest.mark.internet
    def test_URL(self):
        # good url
        url = utils.URL('https://www.dropbox.com/s/qxf6ki0eruv66w5/test.txt?dl=1')
        # __repr__
        print(url)
        # download w/o filename
        url.download('.')
        with Path('test.txt').open('r') as f:
            assert f.readlines() == ['test\n']
        os.remove('./test.txt')
        # download w/ filename
        url.download('.', 'new_test.txt')
        os.remove('./new_test.txt')
        # bad url
        with pytest.raises(ValueError):
            url = utils.URL('https://www.yews.info/BAD_PAGE.html')
