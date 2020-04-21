import torch.nn as nn

from .utils import load_state_dict_from_url

__all__ = ['PolarityV1', 'polarity_v1', 'PolarityCCJ', 'polarity_ccj']

model_urls = {
    'polarity_v1': 'https://www.dropbox.com/s/ckb4glf35agi9xa/polarity_v1_wenchuan-bdd92da2.pth?dl=1',
}

class PolarityV1(nn.Module):

    def __init__(self):
        super().__init__()
        # 600 -> 512
        # n*3*10*600
        # nn.Cov2d(3,16,kernel_size=(3,7),stride=1,padding=....)
        # (3,7) 3>10 7>600 in the end use (3,3).
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=425, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.MaxPool1d(2)
        )

        # 512 -> 256
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.MaxPool1d(2)
        )

        # 256 -> 128
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 128 -> 64
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 64 -> 32
        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 32 -> 16
        self.layer6 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 16 -> 8
        self.layer7 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 8 -> 4
        self.layer8 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 4 -> 2
        self.layer9 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 2 -> 1
        self.layer10 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.fc = nn.Linear(64 * 1, 3)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        print("##### before view #####")
        print(out.shape)
        out = out.view(out.size(0), -1)
        print("##### before fc #####")
        print(out.shape)
        out = self.fc(out)
        print("##### after fc #####")
        print(out.shape)
        return out

def polarity_v1(pretrained=False, progress=True, **kwargs):
    r"""Original CPIC model architecture from the
    `"Deep learning for ..." <https://arxiv.org/abs/1901.06396>`_ paper. The
    pretrained model is trained on 60,000 Wenchuan aftershock dataset
    demonstrated in the paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Wenchuan)
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = PolarityV1(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['polarity_v1'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


class PolarityCCJ(nn.Module):
    r"""a simple recurrent neural network from
    <https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html#creating-the-network>
    """
    def __init__(self, **kwargs):
        super().__init__()
        input_size = 1
        if hidden_size in kwargs:
        hidden_size = 20
        num_layers = 100
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        batch, input_size, seq_len = x.shape
        x = x.permute(2, 0, 1)    # seq_len, batch, input_size
        # If (h_0, c_0) is not provided, both h_0 and c_0 default to zero
        out, (hidden, cell_state) = self.lstm(x)
        # hidden has size [input_size,batch, hidden_size]
        print("###### hidden #####")
        print(hidden.shape)
        x = hidden.permute(1,2,0) 
        x = x.view(x.size(0), -1)
        print("#############")
        out = self.fc(x)
        print("#### out ######")
        print(out.shape)
        print("#############")
        return out

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def polarity_ccj(pretrained=False, progress=True, **kwargs):
    r"""Original CPIC model architecture from the
    `"Deep learning for ..." <https://arxiv.org/abs/1901.06396>`_ paper. The
    pretrained model is trained on 60,000 Wenchuan aftershock dataset
    demonstrated in the paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Wenchuan)
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = PolarityCCJ(**kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['polarity_v1'],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model
