import math
import os
import tarfile
from pathlib import Path
from urllib import request

from torch.utils.model_zoo import tqdm

# dynamically change this limit by Python code.
MEMORY_LIMIT =  2 * 1024 ** 3        # 2 GB limit

def over_memory_limit(path):
    return os.stat(path).st_size > MEMORY_LIMIT

def set_memory_limit(limit):
    global MEMORY_LIMIT
    MEMORY_LIMIT = limit

def get_memory_limit():
    return MEMORY_LIMIT

################################################################################
#
#                           Utilities for URLs
#
################################################################################

def test_url(url):
    try:
        with request.urlopen(url) as req:
            return req
    except:
        return None

def gen_bar_update():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update

def sizeof_fmt(num, suffix='B'):
    """Get human-readable file size

    Args:
        num (int): Number of unit size.
        suffix (str): Unit of size (default: B).

    """
    mag = int(math.floor(math.log(num, 1024)))
    val = num / math.pow(1024, mag)
    if mag > 7:
        return f"{val:.1f}Y{suffix}"
    else:
        return f"{val:3.1f}{['','K','M','G','T','P','E','Z'][mag]}{suffix}"


class URL(object):

    def __init__(self, url):
        self.url = url
        req = test_url(url)
        if req:
            self.req = req
            self.size = req.length
            self.url_filename = self.get_filename()
        else:
            raise ValueError(f"{url} is not a valid URL or unreachable.")

    def get_filename(self):
        return [a.split('=')[1] for a in self.req.info()['content-disposition'].split('; ')
                    if a.startswith('filename=')][0].strip('"')

    def __repr__(self):
        return f"{self.url_filename}, {sizeof_fmt(self.size)}, from <{self.url}>"

    def download(self, root, filename=None):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)    # mkdir if not exists)
        if not filename:
            filename = self.url_filename
        fpath = root / filename

        print(f"Downloading {self.url} to {fpath}")
        request.urlretrieve(self.url, fpath, reporthook=gen_bar_update())


################################################################################
#
#                           Utilities for Bz2
#
################################################################################

def extract_bz2(infile, outdir='.'):
    with tarfile.open(infile, 'r:bz2') as tar:
        tar.extractall(outdir)
