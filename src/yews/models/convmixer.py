from __future__ import annotations

from torch import nn
from torch.nn import Module

__all__ = ["ConvMixer", "convmixer"]


class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixerStem(Module):
    def __init__(
        self,
        in_channels: int,
        dim: int,
        patch_size: int,
        activation=nn.GELU,
    ):
        super().__init__()
        self.patch_embedding = nn.Conv1d(in_channels, dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.act = activation()
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        return self.norm(self.act(self.patch_embedding(x)))


class ConvMixerLayer(Module):
    def __init__(self, dim: int, kernel_size: int, activation=nn.GELU):
        super().__init__()
        self.depth_conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, padding="same", bias=False)
        self.act1 = activation()
        self.norm1 = nn.BatchNorm1d(dim)
        self.point_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.act2 = activation()
        self.norm2 = nn.BatchNorm1d(dim)

    def depth_forward(self, x):
        return self.norm1(self.act1(self.depth_conv(x))) + x

    def point_forward(self, x):
        return self.norm2(self.act2(self.point_conv(x)))

    def forward(self, x):
        return self.point_forward(self.depth_forward(x))


class ClassifierHead(Module):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(in_channels, num_classes) if num_classes > 0 else nn.Identity()


class ConvMixer(Module):
    def __init__(
        self,
        dim: int = 32,
        depth: int = 10,
        kernel_size: int = 9,
        patch_size: int = 7,
        in_channels: int = 3,
        num_classes: int = 3,
        activation=nn.GELU,
        **kwargs,
    ):
        super().__init__()
        self.stem = ConvMixerStem(in_channels, dim, patch_size, activation)
        self.blocks = nn.Sequential(
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv1d(dim, dim, kernel_size, groups=dim, padding="same", bias=False),
                            nn.GELU(),
                            nn.BatchNorm1d(dim),
                        )
                    ),
                    nn.Conv1d(dim, dim, kernel_size=1, bias=False),
                    nn.GELU(),
                    nn.BatchNorm1d(dim),
                )
                for _ in range(depth)
            ],
        )
        self.head = ClassifierHead(dim, num_classes)

    def get_classifier(self):
        return self.head

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


def convmixer(pretrained=False, progress=True, **kwargs):
    model = ConvMixer(**kwargs)
    return model
