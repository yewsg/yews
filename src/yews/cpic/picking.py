import numpy as np
from scipy.signal import find_peaks

from .utils import compute_probs
from .utils import probs2cfs

def pick_arrivals(cf):
    prom = cf.max()
    for i in range(10):
        prom /= 2
        peaks, properties = find_peaks(x=cf, height=0,
                                       distance=10, prominence=prom)

        if peaks.size > 0:
            peak_prom = properties['prominences']
            confidences = peak_prom / peak_prom.sum()
            return peaks, confidences

    return (np.nan, np.nan)


def pick(waveform, fs, wl, model, transform, g=0.1, batch_size=None):
    probs = compute_probs(model, transform, waveform,
                          shape=[3, fs * wl],
                          step=[1, int(g * fs)],
                          batch_size=batch_size)

    # compute cf
    cf_p, cf_s = probs2cfs(probs)

    # find prominent local peaks
    peaks_p, confidences_p = pick_arrivals(cf_p)
    peaks_s, confidences_s = pick_arrivals(cf_s)

    pick_results = {
        'p': peaks_p * g + 5,
        's': peaks_s * g + 5,
        'p_conf': confidences_p,
        's_conf': confidences_s,
        'cf_p': cf_p,
        'cf_s': cf_s
    }

    return pick_results
