import numpy as np
import torch
from obspy import read
from torch import nn

import yews.transforms as tf
from yews.cpic import detect
from yews.cpic import pick
from yews.datasets.utils import stream2array
from yews.models import cpic

st = read('/data/mariana/PA01-Y40/event_p/PA01-Y40off-2017_027_08_20_14.472-m37.BH*')
array = stream2array(st)
model = cpic(pretrained=True)
model = nn.DataParallel(model)

transform = tf.Compose([
    tf.ZeroMean(),
    tf.SoftClip(1e-4),
    tf.ToTensor(),
])

detect_results = detect(array, 100, 20, model, transform, 2, threshold=0.5)
pick_results = pick(array, 100, 20, model, transform, 0.2)

import matplotlib.pyplot as plt
fig, ax1 = plt.subplots()
ts = np.linspace(0, 240, array.shape[1])
ax1.plot(ts, array[2], 'k')
ax2 = ax1.twinx()
td = np.linspace(5, 225, len(detect_results['detect_p']))
tp = np.linspace(5, 225, len(pick_results['cf_p']))
ax2.plot(td, detect_results['detect_p'])
ax2.plot(td, detect_results['detect_s'])
ax2.plot(tp, pick_results['cf_p'])
ax2.plot(tp, pick_results['cf_s'])
plt.show()
