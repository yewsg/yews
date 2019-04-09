import numpy as np

__all__ = [
    "MovingAverageMeter",
    "ExponentialMovingAverageMeter",
    "CumulativeMovingAverageMeter",
]

# -----------------------------------------------------------------------------
#
#   Moving average meters
#
# -----------------------------------------------------------------------------

class MovingAverageMeter(object):

    def __init__(self):
        self.val = None
        self.avg = None
        self.count = 0

    def reset(self, *args):
        raise NotImplementedError

    def update(self, *args):
        raise NotImplementedError


class ExponentialMovingAverageMeter(MovingAverageMeter):
    """Computes and stores the exponential moving average and current value."""

    def __init__(self, alpha=0.5):
        super(ExponentialMovingAverageMeter, self).__init__()
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.count = 0

    def update(self, val, n=1):
        self.count += n
        self.val = val
        for i in range(n):
            if self.avg:
                self.avg = self.alpha * self.val + (1 - self.alpha) * self.avg
            else:
                self.avg = self.val

    def set_alpha(self, alpha):
        self.alpha = alpha


class CumulativeMovingAverageMeter(MovingAverageMeter):
    """Computes and stores the cumulative moving average and current value"""

    def __init__(self):
        super(CumulativeMovingAverageMeter, self).__init__()
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.count = 0

    def update(self, val, n=1):
        if self.avg:
            previous_sum = self.avg * self.count
            self.avg = (previous_sum + n * val) / (self.count + n)
        else:
            self.avg = val

        self.val = val
        self.count += n

