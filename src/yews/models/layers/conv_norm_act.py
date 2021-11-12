from torch import nn
from torch.nn import Module

__all__ = ['ConvNormAct']

class ConvNormAct(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int,
        stride: int = 1,
        padding="",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        norm_layer=nn.BatchNorm1d,
        act_layer=nn.ReLU,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = norm_layer(out_channels)
        self.act = act_layer()

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x
