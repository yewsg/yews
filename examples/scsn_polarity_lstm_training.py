import datetime
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import yews.datasets as dsets
import yews.transforms as transforms
from yews.train import Trainer

#from yews.models import cpic
#from yews.models import cpic_v1
#from yews.models import cpic_v2
#cpic = cpic_v1

from yews.models import polarity_v1
from yews.models import polarity_v2
from yews.models import polarity_lstm
polarity=polarity_lstm


if __name__ == '__main__':

    print("Now: start : " + str(datetime.datetime.now()))

    # Preprocessing
    waveform_transform = transforms.Compose([
        transforms.ZeroMean(),
        #transforms.SoftClip(1e-4),
        transforms.ToTensor(),
    ])

    # Prepare dataset
    dsets.set_memory_limit(10 * 1024 ** 3) # first number is GB
    # dset = dsets.Wenchuan(path='/home/qszhai/temp_project/deep_learning_course_project/cpic', download=False,sample_transform=waveform_transform)
    dset = dsets.SCSN_polarity(path='/home/qszhai/temp_project/deep_learning_course_project/first_motion_polarity/scsn_data/train_npy', download=False, sample_transform=waveform_transform)

    # Split datasets into training and validation
    train_length = int(len(dset) * 0.8)
    val_length = len(dset) - train_length
    train_set, val_set = random_split(dset, [train_length, val_length])

    # Prepare dataloaders
    train_loader = DataLoader(train_set, batch_size=5000, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=10000, shuffle=False, num_workers=4)

    # Prepare trainer
    # trainer = Trainer(cpic(), CrossEntropyLoss(), lr=0.1)
    # note: please use only 1 gpu to run LSTM, https://github.com/pytorch/pytorch/issues/21108
    model_conf = {"hidden_size": 64}
    plt = polarity(**model_conf)
    trainer = Trainer(plt, CrossEntropyLoss(), lr=0.001)

    # Train model over training dataset
    trainer.train(train_loader, val_loader, epochs=50, print_freq=100)
                  #resume='checkpoint_best.pth.tar')

    # Save training results to disk
    trainer.results(path='scsn_polarity_results.pth.tar')

    # Validate saved model
    results = torch.load('scsn_polarity_results.pth.tar')
    #model = cpic()
    model = plt
    model.load_state_dict(results['model'])
    trainer = Trainer(model, CrossEntropyLoss(), lr=0.001)
    trainer.validate(val_loader, print_freq=100)

    print("Now: end : " + str(datetime.datetime.now()))

    import matplotlib.pyplot as plt
    import numpy as np

    myfontsize1=14
    myfontsize2=18
    myfontsize3=24

    results = torch.load('scsn_polarity_results.pth.tar')

    fig, axes = plt.subplots(2, 1, num=0, figsize=(6, 4), sharex=True)
    axes[0].plot(results['val_acc'], label='Validation')
    axes[0].plot(results['train_acc'], label='Training')
    
    #axes[1].set_xlabel("Epochs",fontsize=myfontsize2)
    axes[0].set_xscale('log')
    axes[0].set_xlim([1, 100])
    axes[0].xaxis.set_tick_params(labelsize=myfontsize1)
    
    axes[0].set_ylabel("Accuracies (%)",fontsize=myfontsize2)
    axes[0].set_ylim([0, 100])
    axes[0].set_yticks(np.arange(0, 101, 10))
    axes[0].yaxis.set_tick_params(labelsize=myfontsize1)
    
    axes[0].grid(True, 'both')
    axes[0].legend(loc=4)
    
    #axes[1].semilogx(results['val_loss'], label='Validation')
    #axes[1].semilogx(results['train_loss'], label='Training')
    axes[1].plot(results['val_loss'], label='Validation')
    axes[1].plot(results['train_loss'], label='Training')
    
    axes[1].set_xlabel("Epochs",fontsize=myfontsize2)
    axes[1].set_xscale('log')
    axes[1].set_xlim([1, 100])
    axes[1].xaxis.set_tick_params(labelsize=myfontsize1)
    
    axes[1].set_ylabel("Losses",fontsize=myfontsize2)
    axes[1].set_ylim([0.0, 1.0])
    axes[1].set_yticks(np.arange(0.0,1.01,0.2))
    axes[1].yaxis.set_tick_params(labelsize=myfontsize1)
    
    axes[1].grid(True, 'both')
    axes[1].legend(loc=1)
    
    fig.tight_layout()
    plt.savefig('Accuracies_train_val.pdf') 
