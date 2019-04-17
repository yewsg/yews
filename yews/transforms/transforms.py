from .base import BaseTransform
from . import functional as F

__all__ = [
    "ToTensor",
    "ToInt",
    "ZeroMean",
    "SoftClip",
    "CutWaveform",
]

class ToTensor(BaseTransform):
    """Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (C x S) to a torch.FloatTensor of shape (C x S).
    """

    def __call__(self, wav):
        """
        Args:
            wav: Waveform to be converted to tensor.
        Returns:
            Tensor: Converted tensor.
        """
        return F._to_tensor(wav)


class ToInt(BaseTransform):
    """Convert a label to int based on the given lookup table.

    Args:
        lookup (dict): Lookup table to convert a label to int.

    """

    def __init__(self, lookup):
        if type(lookup) is dict:
            self.lookup = lookup
        else:
            raise ValueError("Lookup table needs to be a dictionary.")
        if any([type(val) is not int for val in self.lookup.values()]):
            raise ValueError("Values of the lookup table need to be Int.")

    def __call__(self, label):
        return self.lookup[label]


class SoftClip(BaseTransform):
    """Soft clip input to compress large amplitude signals

    """

    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, wav):
        return F.expit(wav * self.scale)


class ZeroMean(BaseTransform):
    """Remove mean from each waveforms

    """

    def __call__(self, wav):
        wav = wav.astype(float).T
        wav -= wav.mean(axis=0)
        return wav.T


class CutWaveform(BaseTransform):
    """Cut a portion of waveform.

    """

    def __init__(self, samplestart, sampleend):
        self.start = int(samplestart)
        self.end = int(sampleend)

    def __call__(self, wav):
        return wav[:, self.start:self.end]

