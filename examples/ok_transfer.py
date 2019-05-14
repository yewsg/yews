from yews.dataset import DatasetFolder

class TransferOK(DatasetFolder):
    def build_dataset(self):
        """Return samples and targets.

        """
        files = self.get_files_under_dir(self.root, '**/*')
        labels = [p.name.split('.')[2] for p in files]
        samples = self.FilesLoader(files, self.loader)

        return samples, labels

if __name__ == '__main__':
    # Preprocessing
    waveform_transform = transforms.Compose([
        transforms.ZeroMean(),
        transforms.SoftClip(1e-4),
        transforms.ToTensor(),
    ])

    dset = TransferOK(path='/home/chenyu/wenchuan/src/data/ok_transfer')

    print(len(dset))
