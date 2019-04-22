from .dirs import DatasetArrayFolder
from .utils import extract_bz2
from .utils import URL

class Wenchuan(DatasetArrayFolder):
    """`Wenchuan <https://arxiv.org/abs/1901.06396>`_ Dataset.

    Args:
        path (str): Root directory where ``wenchuan/processed/data.npy`
            and ``wenchuan/processed/labels.npy`` exists.
        download (bool, optional): If True, downloads the dataset from internet
            and puts it in root directory. If dataset is already downloaded, it
            will not be downloaded again.
        sample_transform (callable, optional): A function/transform that takes
            a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            a target and transform it.

    """

    url = URL('https://www.dropbox.com/s/enr75zt2ukx118r/wenchuan.tar.bz2?dl=1')

    def __init__(self, path, download=False, **kwargs):
        # verify download flag
        if type(download) is not bool:
            raise ValueError("`download` needs to be True or False.")

        # verify if dataset is ready
        try:
            super().__init__(path=path, **kwargs)
        except ValueError:
            if download:
                # download compressed file from source if not exists
                fpath = self.root / self.url.url_filename
                if not fpath.is_file():
                    self.url.download(self.root)
                # extract file under root directory
                print("Extracting dataset ...")
                extract_bz2(fpath, self.root)
                # try initiate DatasetArrayFolder again
                super().__init__(path=path, **kwargs)
            else:
                raise ValueError(f"{self.root} contains no valid dataset. "
                                 f"Consider set `download=True` and remove broken bz2 file.")

    def download(self):
        for name, url in self.urls.items():
            download_url(url, self.root, filename=name)
