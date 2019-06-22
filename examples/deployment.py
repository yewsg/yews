import matplotlib.pyplot as plt
import numpy as np
import obspy
import torch
from torch import nn

import yews.transforms as tf
from yews.cpic import detect
from yews.cpic import pick
from yews.datasets.utils import stream2array
from yews.models import cpic

origin_time = obspy.UTCDateTime('2019-05-14T12:58:26.000000Z')
st = obspy.read('/data/sp.mseed',
                starttime=origin_time,
                endtime=origin_time + 100)
array = stream2array(st)
model = cpic(pretrained=True)
model = nn.DataParallel(model)

transform = tf.Compose([
    tf.ZeroMean(),
    tf.SoftClip(1e-6),
    tf.ToTensor(),
])

fig, axes = plt.subplots(4, 1, figsize=(6, 8), sharex=True)

pick_results = pick(array, 100, 20, model, transform, 0.5)
ts = st[0].times()
ax1 = axes[0]
ax1.plot(ts, array[2], 'k')
ax1.set_ylabel('Amplitude (counts)')
ax2 = ax1.twinx()
tp = np.linspace(5, ts[-1]-15, len(pick_results['cf_p']))
ax2.plot(tp, pick_results['cf_p'])
ax2.plot(tp, pick_results['cf_s'])
ax2.set_xlim(0, 100)
ax2.grid()
ax2.set_ylabel('Probability ratio (log10)')

st = obspy.read('/data/sp.mseed',
                starttime=origin_time + 1000,
                endtime=origin_time + 1100)
array = stream2array(st)
model = cpic(pretrained=True)
model = nn.DataParallel(model)

transform = tf.Compose([
    tf.ZeroMean(),
    tf.SoftClip(1e-6),
    tf.ToTensor(),
])

pick_results = pick(array, 100, 20, model, transform, 0.5)
ax1 = axes[1]
ax1.plot(ts, array[2], 'k')
ax1.set_ylabel('Amplitude (counts)')
ax2 = ax1.twinx()
tp = np.linspace(5, ts[-1]-15, len(pick_results['cf_p']))
ax2.plot(tp, pick_results['cf_p'])
ax2.plot(tp, pick_results['cf_s'])
ax2.set_xlim(0, 100)
ax2.grid()
ax2.set_ylabel('Probability ratio (log10)')

st = obspy.read('/data/sp.mseed',
                starttime=origin_time + 2000,
                endtime=origin_time + 2100)
array = stream2array(st)
model = cpic(pretrained=True)
model = nn.DataParallel(model)

transform = tf.Compose([
    tf.ZeroMean(),
    tf.SoftClip(1e-6),
    tf.ToTensor(),
])

pick_results = pick(array, 100, 20, model, transform, 0.5)
ax1 = axes[2]
ax1.plot(ts, array[2], 'k')
ax1.set_ylabel('Amplitude (counts)')
ax2 = ax1.twinx()
tp = np.linspace(5, ts[-1]-15, len(pick_results['cf_p']))
ax2.plot(tp, pick_results['cf_p'])
ax2.plot(tp, pick_results['cf_s'])
ax2.set_xlim(0, 100)
ax2.grid()
ax2.set_ylabel('Probability ratio (log10)')

st = obspy.read('/data/sp.mseed',
                starttime=origin_time + 3000,
                endtime=origin_time + 3100)
array = stream2array(st)
model = cpic(pretrained=True)
model = nn.DataParallel(model)

transform = tf.Compose([
    tf.ZeroMean(),
    tf.SoftClip(1e-4),
    tf.ToTensor(),
])

pick_results = pick(array, 100, 20, model, transform, 0.5)
ax1 = axes[3]
ax1.plot(ts, array[2], 'k')
ax1.set_ylim(-100000, 100000)
ax1.set_ylabel('Amplitude (counts)')
ax2 = ax1.twinx()
tp = np.linspace(5, ts[-1]-15, len(pick_results['cf_p']))
ax2.plot(tp, pick_results['cf_p'])
ax2.plot(tp, pick_results['cf_s'])
ax2.set_xlim(0, 100)
ax2.grid()
ax2.set_ylabel('Probability ratio (log10)')

ax1.set_xlabel('Time (second)')
fig.tight_layout()

fig, ax = plt.subplots(1,1, figsize=(10, 4))
st = obspy.read('/data/sp.mseed',
                starttime=origin_time,
                endtime=origin_time + 7200)
ts = st[0].times()
array = stream2array(st)
model = cpic(pretrained=True)
model = nn.DataParallel(model)

transform = tf.Compose([
    tf.ZeroMean(),
    tf.SoftClip(1e-4),
    tf.ToTensor(),
])

pick_results = pick(array, 100, 20, model, transform, 0.5)
ax1 = ax
ax1.plot(ts, array[2], 'k')
ax1.set_ylabel('Amplitude (counts)')
ax2 = ax1.twinx()
tp = np.linspace(5, ts[-1]-15, len(pick_results['cf_p']))
ax2.plot(tp, pick_results['cf_p'])
ax2.plot(tp, pick_results['cf_s'])
ax2.set_xlim(0, 7200)
ax2.grid()
ax2.set_ylabel('Probability ratio (log10)')

ax1.set_xlabel('Time (second)')
fig.tight_layout()
plt.show()
