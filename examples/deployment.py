import numpy as np
import obspy
import torch
from torch import nn

import yews.transforms as tf
from yews.cpic import detect
from yews.cpic import pick
from yews.datasets.utils import stream2array
from yews.models import cpic

st = obspy.read('/data/sp.mseed',
                starttime=obspy.UTCDateTime('2019-05-14T12:58:26.000000Z'),
                endtime=obspy.UTCDateTime('2019-05-14T16:08:00.000000Z'))
array = stream2array(st)
model = cpic(pretrained=True)
model = nn.DataParallel(model)

transform = tf.Compose([
    tf.ZeroMean(),
    tf.SoftClip(1e-6),
    tf.ToTensor(),
])

#detect_results = detect(array, 100, 20, model, transform, 2, threshold=0.5)
pick_results = pick(array, 100, 20, model, transform, 0.5)
#
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1)
ts = st[0].times()
axes[0].plot(ts, array[2], 'k')
ax2 = axes[0].twinx()
#td = np.linspace(5, ts[-1]-15, len(detect_results['detect_p']))
#ax2.plot(td, detect_results['detect_p'])
#ax2.plot(td, detect_results['detect_s'])
tp = np.linspace(5, ts[-1]-15, len(pick_results['cf_p']))
ax2.plot(tp, pick_results['cf_p'])
ax2.plot(tp, pick_results['cf_s'])
plt.show()
