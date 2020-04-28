import datetime
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import yews.datasets as dsets
import yews.transforms as transforms
from yews.train import Trainer

#from yews.models import cpic
from yews.models import cpic_v1
from yews.models import cpic_v2
from yews.models import cpic_v3
cpic = cpic_v3

if __name__ == '__main__':

    print("Now: start : " + str(datetime.datetime.now()))

    # Preprocessing
    waveform_transform = transforms.Compose([
        transforms.ZeroMean(),
        transforms.SoftClip(1e-3),
        transforms.ToTensor(),
    ])

    # Prepare dataset
    dsets.set_memory_limit(10 * 1024 ** 3) # first number is GB
    dset = dsets.Wenchuan(path='/home/qszhai/temp_project/deep_learning_course_project/cpic/wenchuan_data/train_npy', download=False,sample_transform=waveform_transform)

    # Split datasets into training and validation
    train_length = int(len(dset) * 0.8)
    val_length = len(dset) - train_length
    train_set, val_set = random_split(dset, [train_length, val_length])

    # Prepare dataloaders
    train_loader = DataLoader(train_set, batch_size=1000, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=2000, shuffle=False, num_workers=4)

    # Prepare trainer
    trainer = Trainer(cpic(), CrossEntropyLoss(), lr=0.01)

    # Train model over training dataset
    trainer.train(train_loader, val_loader, epochs=100, print_freq=10)
                  #resume='checkpoint_best.pth.tar')

    # Save training results to disk
    trainer.results(path='wenchuan_results.pth.tar')

    # Validate saved model
    results = torch.load('wenchuan_results.pth.tar')
    model = cpic()
    model.load_state_dict(results['model'])
    trainer = Trainer(model, CrossEntropyLoss(), lr=0.1)
    trainer.validate(val_loader, print_freq=100)

    print("Now: end : " + str(datetime.datetime.now()))
