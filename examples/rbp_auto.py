import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import yews.transforms as tf
from yews.cpic import pick
from yews.cpic.utils import sliding_window_view
from yews.datasets.utils import stream2array
from yews.models import cpic_v1
from yews.models import cpic_v2

t0 = time.time()
waveform = np.load('rbp.npy')
print(f"Loading data takes {time.time() - t0} seconds.")
t0 = time.time()
model = cpic_v1(pretrained=False)
with open('cpic_model.pickle', 'rb') as f:
    state_dict = pickle.load(f)
model.load_state_dict(state_dict)
model = cpic_v2(pretrained=True)
model.eval()
print(f"Loading model takes {time.time() - t0} seconds.")

transform = tf.Compose([
    tf.ZeroMean(),
    tf.SoftClip(1e-3),
    tf.ToTensor(),
])

t0 = time.time()
pick_results = pick(waveform, 100, 20, model, transform, 0.5, 15)
print(f"Picking takes {time.time() - t0} seconds.")
cf_p = pick_results['cf_p']
cf_s = pick_results['cf_s']

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
# Waveform
ax = axes[0]
tw = np.linspace(0, 240, waveform.shape[1])
waveform = (waveform.T - waveform.T.mean(axis=0)).T
scale = np.abs(waveform).max()
waveform /= scale
for i, tr in enumerate(waveform):
    ax.plot(tw, tr - i, 'k', alpha=0.8)
ax.axvline(x=120, color='b', linestyle='-', label='P catalog')
ax.axvline(x=134, color='g', linestyle='-', label='S catalog')
ax.set_yticks([-2, -1, 0])
ax.set_yticklabels(['BHZ', 'BHN', 'BHE'])
ax.grid(True)
ax.set_ylabel('Waveforms')
ax.legend(loc=1)
# outputs
ax = axes[1]
td = np.linspace(5, 225, cf_p.shape[0])
ax.plot(td, cf_p, 'b:', label='P CF')
ax.plot(td, cf_s, 'g:', label='S CF')
ax.axvline(x=120, color='b', linestyle='-', label='P catalog')
ax.axvline(x=134, color='g', linestyle='-', label='S catalog')
ax.set_xlim([0, 240])
ax.set_xlabel('Time (s)')
ax.set_ylabel('CF')
ax.grid(True)
ax.legend()

fig.tight_layout()
plt.show()
#fig.savefig('cpic_demo.pdf', transparent=True)
