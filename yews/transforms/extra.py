try:
    from scipy.special import expit
    no_scipy = False
except ModuleNotFoundError:
    no_scipy = True

from .base import BaseTransform

__all__ = [
    "ExtraTransform",
    "SoftClip",
]

class ExtraTransform(BaseTransform):

    def __init__(self):
        if no_scipy:
            raise ModuleNotFoundError(f"Install Scipy to use the extra transform ({self.__class__.__name__})")


class SoftClip(ExtraTransform):
    """Soft clip input to compress large amplitude signals.

    """

    def __init__(self, scale=1):
        super(SoftClip, self).__init__()
        self.scale = scale

    def __call__(self, wav):
        return expit(wav * self.scale)

