def is_transform(obj):
    """Verfy if a object is a ``transform-like`` object.

    Args:
        obj: Object to be determined.

    Returns:
        bool: True for ``transform-like`` object, false otherwise.

    """
    return hasattr(obj, '__call__')

class BaseTransform(object):
    """An abstract class representing a Transform.

    All other transform should subclass it. All subclasses should override
    ``__call__`` which performs the transform.

    Note:
        A transform-like object has ``__call__`` implmented. Typical
        transform-like objects include python functions and methods.

    """

    def __call__(self, data):
        raise NotImplementedError

    def __repr__(self):
        head = self.__class__.__name__
        content = [f"{key} = {val}" for key, val in self.__dict__.items()]
        body = ", ".join(content)
        return f"{head}({body})"


class Compose(BaseTransform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.ZeroMean(),
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
