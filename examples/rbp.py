import pickle
import time

import numpy as np
from scipy.special import expit
from scipy.ndimage.filters import gaussian_filter1d
import torch
from torch import nn
import matplotlib.pyplot as plt

import yews.transforms as tf
from yews.cpic.utils import sliding_window_view
from yews.datasets.utils import stream2array
from yews.models import cpic

t0 = time.time()
waveform = np.load('rbp.npy')
print(f"Loading data takes {time.time() - t0} seconds.")
print(waveform.shape)
t0 = time.time()
model = cpic(pretrained=False)
with open('cpic_model.pickle', 'rb') as f:
    state_dict = pickle.load(f)
model.load_state_dict(state_dict)
model.eval()
print(f"Loading model takes {time.time() - t0} seconds.")

transform = tf.Compose([
    tf.ZeroMean(),
    tf.SoftClip(1e-3),
    tf.ToTensor(),
])

t0 = time.time()
windows = np.squeeze(sliding_window_view(waveform, [3, 2000], [1, 20]))
windows = torch.stack([transform(window) for window in windows])
print(f"Breaking into windows takes {time.time() - t0} seconds.")
print(windows.shape)

t0 = time.time()
outputs = []
batch = 15
with torch.no_grad():
    for i in range(0, 1000, batch):
        outputs.append(model(windows[i:(i+batch)]))
outputs = np.concatenate(outputs, axis=0)
print(f"Computing outputs on windows takes {time.time() - t0} seconds.")
print(outputs.shape)

# convert outputs to probabilities
probs = expit(outputs).T
probs /= probs.sum(axis=0)

# convert probabilities to cfs
cf_p = np.log10(probs[1] / (probs[0] + 1e-5))
cf_s = np.log10(probs[2] / (probs[0] + 1e-5))
cf_p[probs.argmax(axis=0) != 1] = 0
cf_s[probs.argmax(axis=0) != 2] = 0
sigma=3
cf_p = gaussian_filter1d(cf_p, sigma=sigma)
cf_s = gaussian_filter1d(cf_s, sigma=sigma)
print(cf_p.shape)

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
td = np.linspace(-22, 225, cf_p.shape[0])
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
