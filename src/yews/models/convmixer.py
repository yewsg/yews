from torch import nn

__all__ = ["ConvMixer", "convmixer"]


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=3):
    return nn.Sequential(
        nn.Conv1d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm1d(dim),
        *[
            nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv1d(dim, dim, kernel_size, groups=dim, padding="same"),
                        nn.GELU(),
                        nn.BatchNorm1d(dim),
                    )
                ),
                nn.Conv1d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm1d(dim),
            )
            for _ in range(depth)
        ],
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(dim, n_classes),
    )


def convmixer(pretrained=False, progress=True, **kwargs):
    model = ConvMixer(32, 10, n_classes=3)
    return model
