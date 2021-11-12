from __future__ import annotations

from torch import nn
from torch.nn import Module
from .layers import ClassifierHead

__all__ = ["ConvMixer", "convmixer_64_20_9_14"]


class ConvMixerStem(Module):
    def __init__(
        self,
        in_channels: int,
        dim: int,
        patch_size: int,
        activation=nn.GELU,
    ):
        super().__init__()
        self.patch_embedding = nn.Conv1d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.act = activation()
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        return self.norm(self.act(self.patch_embedding(x)))


class ConvMixerLayer(Module):
    def __init__(self, dim: int, kernel_size: int, norm_layer=nn.BatchNorm1d, act_layer=nn.GELU):
        super().__init__()
        self.depth_conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, padding="same")
        self.act1 = act_layer()
        self.norm1 = norm_layer(dim)
        self.point_conv = nn.Conv1d(dim, dim, kernel_size=1)
        self.act2 = act_layer()
        self.norm2 = norm_layer(dim)

    def depth_forward(self, x):
        return self.norm1(self.act1(self.depth_conv(x))) + x

    def point_forward(self, x):
        return self.norm2(self.act2(self.point_conv(x)))

    def forward(self, x):
        return self.point_forward(self.depth_forward(x))


class ConvMixer(Module):
    def __init__(
        self,
        dim: int = 32,
        depth: int = 10,
        kernel_size: int = 9,
        patch_size: int = 7,
        in_channels: int = 3,
        num_classes: int = 3,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.stem = ConvMixerStem(in_channels, dim, patch_size, act_layer)
        for i in range(depth):
            self.add_module(f"b{i+1}", ConvMixerLayer(dim, kernel_size, act_layer))
        self.head = ClassifierHead(dim, num_classes)

    def forward_features(self, x):
        for block in list(self.children())[:-1]:
            x = block(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


def convmixer_64_20_9_14(pretrained=False, progress=True, **kwargs):
    model = ConvMixer(64, 20, 9, 14)
    return model
