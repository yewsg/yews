from torch import nn
from torch.nn import Module

__all__ = ['ClassifierHead']

class ClassifierHead(Module):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(in_channels, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
