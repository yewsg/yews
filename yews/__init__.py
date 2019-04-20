import pkg_resources
__version__ = pkg_resources.get_distribution('yews').version

from yews import datasets
from yews import transforms
from yews import train
from yews import models
