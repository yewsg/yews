import os
from pathlib import Path

import pytest

from yews.datasets import utils

class TestBz2Utils():

    def test_bz2_extraction(self):
        utils.extract_bz2('tests/assets/test.tar.bz2')
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
