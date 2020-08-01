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
        #transforms.RemoveTrend(),
        #transforms.RemoveMean(),
        #transforms.Taper(),
        #transforms.BandpassFilter(),
        #transforms.SoftClip(2e-3),
        #1e-2=1/100 100=1% max
        #2e-3=4/2048  hist: max = 2048
        #import numpy as np;import matplotlib.pyplot as plt;samples=np.load("samples.npy",mmap_mode='r');
        #targets=np.load("targets.npy");target.shape
        #plt.hist(samples[0:100000,0,:].flatten(), bins=100); plt.ylim([0.1,1.5e8]);plt.show()
        transforms.ToTensor(),
    ])

    # Prepare dataset
    dsets.set_memory_limit(10 * 1024 ** 3) # first number is GB
    dset = dsets.Taiwan20092010(path='/home/qszhai/temp_project/deep_learning_course_project/cpic/Taiwan20092010', download=False, sample_transform=waveform_transform)

    # Split datasets into training and validation
    train_length = int(len(dset) * 0.8)
    val_length = len(dset) - train_length
    train_set, val_set = random_split(dset, [train_length, val_length])

    # Prepare dataloaders
    train_loader = DataLoader(train_set, batch_size=2000, shuffle=True, num_workers=4)
    # train_set: bastch_size = targets.shape / 500
    val_loader = DataLoader(val_set, batch_size=4000, shuffle=False, num_workers=4)
    # train_set: bastch_size : larger is better if the GPU memory is enough.
    # num_workers = number of cpu core, but limited by the disk speed. so 8 is good.

    # Prepare trainer
    trainer = Trainer(cpic(), CrossEntropyLoss(), lr=0.1)

    # Train model over training dataset
    trainer.train(train_loader, val_loader, epochs=300, print_freq=100)
                  #resume='checkpoint_best.pth.tar')

    # Save training results to disk
    trainer.results(path='Taiwan20092010_results.pth.tar')

    # Validate saved model
    results = torch.load('Taiwan20092010_results.pth.tar')
    model = cpic()
    model.load_state_dict(results['model'])
    trainer = Trainer(model, CrossEntropyLoss(), lr=0.1)
    trainer.validate(val_loader, print_freq=100)

    print("Now: end : " + str(datetime.datetime.now()))
