from torch import nn
from torch.nn import Module
from .layers import ConvNormAct, ClassifierHead


class Cpic(Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        kernel_size: int,
        patch_size: int = 7,
        in_channels: int = 3,
        num_classes: int = 3,
        act_layer=nn.ReLU,
    ):
        super().__init__()
        self.stem = ConvNormAct(
            in_channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
            act_layer=act_layer,
        )
        for i in range(depth):
            self.add_module(
                f"b{i+1}", ConvNormAct(dim, dim, kernel_size, stride=2, padding="valid", act_layer=act_layer)
            )
        self.head = ClassifierHead(dim, num_classes)

    def forward_features(self, x):
        for block in list(self.children())[:-1]:
            x = block(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


def cpic(pretrained=False, progress=True, **kwargs):
    model = Cpic(64, 8, 3, patch_size=5)
    return model
