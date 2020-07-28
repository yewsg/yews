from . import functional as F
from .base import BaseTransform
<<<<<<< HEAD
from scipy import signal
=======
>>>>>>> upstream/master

try:
    from scipy.special import expit
except ModuleNotFoundError:
    # fake expit if scipy is not installed
    import numpy as np
    expit = lambda x: 1 / (1 + np.exp(-x))

__all__ = [
    "ToTensor",
    "ToInt",
    "Select",
    "ZeroMean",
    "CutWaveform",
    "SoftClip",
<<<<<<< HEAD
    "RemoveMean",
    "RemoveTrend",
    "Taper",
    "BandpassFilter",
=======
>>>>>>> upstream/master
]

class ToTensor(BaseTransform):
    """Converts a numpy.ndarray C x S) to a torch.FloatTensor of shape (C x S).

    """

    def __call__(self, wav):
        return F._to_tensor(wav)


class ToInt(BaseTransform):
    """Convert a label to int based on the given lookup table.

    Args:
        lookup (dict): Lookup table to convert a label to int.

    """

    def __init__(self, lookup):
        if isinstance(lookup, dict):
            self.lookup = lookup
        else:
            raise ValueError("Lookup table needs to be a dictionary.")
        if any([not isinstance(val, int) for val in self.lookup.values()]):
            raise ValueError("Values of the lookup table need to be Int.")

    def __call__(self, label):
        return self.lookup[label]


class Select(BaseTransform):
    """Select an item from the iterable of each datapoint.

    Args:
        index (int): Index to select the item.
    """

    def __init__(self, index):
        if isinstance(index, int):
            self.index = index
        else:
            raise ValueError("Index needs to be a int.")

    def __call__(self, label):
        return label[self.index]


class SoftClip(BaseTransform):
    """Soft clip input to compress large amplitude signals.

    """

    def __init__(self, scale=1.):
        if isinstance(scale, float) or isinstance(scale, int):
            self.scale = scale
        else:
            raise ValueError("Scale needs to be a float.")

    def __call__(self, wav):
        return expit(wav * self.scale)


class ZeroMean(BaseTransform):
    """Remove mean from each waveforms.

    """

    def __call__(self, wav):
        wav = wav.astype(float).T
        wav -= wav.mean(axis=0)
        return wav.T


class CutWaveform(BaseTransform):
    """Cut a portion of the input waveform.

    """

    def __init__(self, samplestart, sampleend):
        self.start = int(samplestart)
        self.end = int(sampleend)

    def __call__(self, wav):
        return wav[:, self.start:self.end]
<<<<<<< HEAD

class RemoveMean(BaseTransform):
    """Remove mean from each waveforms.

    """
    def __call__(self, wav):
        [x,y] = wav.shape
        wav = wav - wav.mean(axis=-1).reshape((x,1))
        return wav

class RemoveTrend(BaseTransform):
    """Remove trend from each waveforms.

    """
    def __call__(self, wav):
        wav = signal.detrend(wav,axis=-1)
        return wav

class Taper(BaseTransform):
    """Add taper in both ends of each waveforms.

    """
    def __call__(self, wav):

        half_taper = 0.05

        [x,y] = wav.shape
        tukey_win = signal.tukey(y, alpha=2*half_taper, sym=True)
        wav = wav * tukey_win
        return wav

class BandpassFilter(BaseTransform):
    """Apply Bandpass filter to each waveforms.
    """
    def __call__(self, wav):

        delta = 0.01 #s, delta in sac file header. it is 1/sampling_rate
        order = 4
        lowfreq = 2 # Hz
        highfreq = 16 # Hz

        nyq = 0.5 * (1 / delta)
        low = lowfreq / nyq
        high = highfreq / nyq
        b, a = signal.butter(order, [low, high], btype='bandpass')
        wav = signal.filtfilt(b, a, wav, axis=-1, padtype=None, padlen=None, irlen=None)

        return wav
=======
>>>>>>> upstream/master
