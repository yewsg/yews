from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import yews.transforms as transforms
from yews.datasets import DatasetArray
from yews.models import Cpic
from yews.train import Trainer


if __name__ == '__main__':
    # Preprocessing
    waveform_transform = transforms.Compose([
        transforms.ZeroMean(),
        transforms.CutWaveform(500, 2500),
        transforms.SoftClip(1e-4),
        transforms.ToTensor(),
    ])
    lookup = {
        'N': 0,
        'P': 1,
        'S': 2,
    }
    target_transform = transforms.ToInt(lookup)

    # Prepare dataset
    wenchuan_dataset = DatasetArray(root='/data/wenchuan/samples.npy',
                                    sample_transform=waveform_transform,
                                    target_transform=target_transform)

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
