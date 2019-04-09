import numpy as np

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import SubsetRandomSampler

import yews.transforms as transforms
from yews.dataset import ClassificationDatasetArray
from yews.train import Trainer
from yews.models import Cpic2


class WenchuanTrainer(Trainer):

    def _update_scheduler(self):
        self.scheduler.step(self.val_loss[-1])


if __name__ == '__main__':
    # Preprocessing
    waveform_transform = transforms.Compose([
        transforms.ZeroMean(),
        transforms.CutWaveform(500, 2500),
        transforms.SoftClip(1e-4),
        transforms.ToTensor(),
    ])
    # Prepare dataset
    samples = np.load('/data/wenchuan/samples.npy')
    wenchuan_dataset = ClassificationDatasetArray(samples,
                                                  transform=waveform_transform,
                                                  target_transform=None)

    # Split datasets into training and validation
    ratio = 0.8
    #sample_list = np.arange(len(wenchuan_dataset))
    sample_list = np.random.permutation(len(wenchuan_dataset))
    split_point = int(0.8 * len(wenchuan_dataset))
    train_list = sample_list[:split_point]
    val_list = sample_list[split_point:]

    # Prepare training
    trainer = WenchuanTrainer(wenchuan_dataset, Cpic2, CrossEntropyLoss(),
                              Adam, ReduceLROnPlateau, lr=0.1)
    train_loader = trainer.prepare_dataloader(
        batch_size=100,
        sampler=SubsetRandomSampler(train_list),
        num_workers=4
    )
    val_loader = trainer.prepare_dataloader(
        batch_size=1000,
        sampler=SubsetRandomSampler(val_list),
        num_workers=4
    )

    trainer.train(train_loader, val_loader, epochs=130, print_freq=100)
