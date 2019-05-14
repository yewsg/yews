from obspy import read

from yews.datasets import DatasetFolder

class TransferOK(DatasetFolder):
    def build_dataset(self):
        """Return samples and targets.

        """
        path = self.root
        files = [p for p in path.glob('*/*') if p.is_file()]
        labels = [p.name.split('.')[2] for p in files]
        samples = self.FilesLoader(files, self.loader)

        return samples, labels

if __name__ == '__main__':
    dset = TransferOK(path='/home/chenyu/wenchuan/src/data/ok_transfer',
                      loader=read)

    print(len(dset))
