from . import functional as F

__all__ = [
    "Compose",
    "ToTensor",
]

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, wav):
        for t in self.transforms:
            wav = t(wav)
        return wav

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
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
        return F.to_tensor(wav)

    def __repr__(self):
        return self.__class__.__name__ + '()'

