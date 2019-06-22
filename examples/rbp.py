import pickle

import numpy as np
import torch
from torch import nn

import yews.transforms as tf
from yews.cpic.utils import sliding_window_view
from yews.datasets.utils import stream2array
from yews.models import cpic

array = np.load('rbp.npy')
model = cpic()
with open('test.pickle', 'rb') as f:
    state_dict = pickle.load(f)
model.load_state_dict(state_dict)
model.eval()

transform = tf.Compose([
    tf.ZeroMean(),
    tf.SoftClip(1e-6),
    tf.ToTensor(),
])

windows = np.squeeze(sliding_window_view(array, [3, 2000], [1, 20]))
windows = torch.stack([transform(window) for window in windows])

with torch.no_grad():
    for i in range(0, 1000, 10):
        print(model(windows[i:(i+10)]))
