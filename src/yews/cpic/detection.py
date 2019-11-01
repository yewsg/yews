import numpy as np
import torch
from scipy.special import expit

from .utils import compute_probs
from .utils import sliding_window_view

def find_nonzero_runs(a):
    # source: https://stackoverflow.com/
    # questions/31544129/extract-separate-non-zero-blocks-from-array

    # Create an array that is 1 where a is nonzero, and pad each end with an extra 0.
    isnonzero = np.concatenate(([0], (np.asarray(a) != 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isnonzero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges - [0, 1]


def detect(waveform, fs, wl, model, transform, g, threshold=0.5,
           batch_size=None):
    probs = compute_probs(model, transform, waveform,
                          shape=[3, fs * wl],
                          step=[1, int(g * fs)],
                          batch_size=batch_size)

    probs[probs < threshold] = 0
    p_prob, s_prob = probs[1:]

    # detect window length
    p = find_nonzero_runs(p_prob)
    s = find_nonzero_runs(s_prob)

    detect_results = {
        'p': p * g + 5,
        's': s * g + 5,
        'detect_p': p_prob,
        'detect_s': s_prob,
    }

    return detect_results
