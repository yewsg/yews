import numpy as np

from torch.nn import CrossEntropyLoss
from torch.utils.data import random_split, DataLoader

import yews.transforms as transforms
from yews.dataset import ClassificationDatasetArray as CDA
from yews.train import Trainer
from yews.models import Cpic


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
    wenchuan_dataset = CDA(samples, transform=waveform_transform)

    # Split datasets into training and validation
    train_length = int(len(wenchuan_dataset) * 0.8)
    val_length = len(wenchuan_dataset) - train_length
    train_set, val_set = random_split(wenchuan_dataset, [train_length, val_length])

    # Prepare dataloaders
    train_loader = DataLoader(train_set, batch_size=100, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=1000, shuffle=False, num_workers=8)

    # Prepare trainer
    trainer = Trainer(Cpic, CrossEntropyLoss(), lr=0.1)

    # Train model over training dataset
    trainer.train(train_loader, val_loader, epochs=30, print_freq=100)
