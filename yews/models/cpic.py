import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'Cpic',
]

class Cpic(nn.Module):

    def __init__(self):
        super(Cpic, self).__init__()
        # 2000 -> 1000
        self.layer1 = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=5, stride=1, padding=26, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.MaxPool1d(2)
        )

        # 1000 -> 500
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.MaxPool1d(2)
        )

        # 500 -> 250
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 250 -> 125
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 126 -> 63
        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 63 -> 31
        self.layer6 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 31 -> 15
        self.layer7 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 15 -> 7
        self.layer8 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 7 -> 3
        self.layer9 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 3 -> 1
        self.layer10 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 3 -> 2
        self.layer11 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # # 3 -> 2
        # self.layer11 = nn.Sequential(
        #     nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2)
        # )

        self.fc = nn.Linear(64 * 1, 3)

    def forward(self, x):
        # print(x.size())
        out = self.layer1(x)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = self.layer4(out)
        # print(out.size())
        out = self.layer5(out)
        # print(out.size())
        out = self.layer6(out)
        # print(out.size())
        out = self.layer7(out)
        # print(out.size())
        out = self.layer8(out)
        # print(out.size())
        out = self.layer9(out)
        # print(out.size())
        out = self.layer10(out)
        # print(out.size())
        out = self.layer11(out)
        # print(out.size())
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)

        return out

