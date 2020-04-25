import torch
import torch.nn as nn

from .utils import load_state_dict_from_url

__all__ = ['PolarityV1', 'polarity_v1', 'PolarityLSTM', 'polarity_lstm']

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
        print("##### input to forward #####")
        print(x.shape)

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
        out = out.view(out.size(0), -1)
        out = self.fc(out)
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


class PolarityLSTM(nn.Module):
    r"""a LSTM neural network
    @author: Chujie Chen
    @Email: chen8chu8jie6@gmail.com
    @date: 04/24/2020
    """
    def __init__(self, **kwargs):
        super().__init__()
        input_size = 1
        self.hidden_size = kwargs["hidden_size"]
        self.bidirectional = kwargs["bidirectional"]
        self.contains_unkown = kwargs["contains_unkown"]
        self.start = kwargs['start']
        self.end = kwargs['end']
        self.num_layers = kwargs['num_layers']
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, 
                            bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), 3 if self.contains_unkown else 2)

    def forward(self, x):
        x = x.narrow(2,self.start, self.end-self.start)
        x = x.permute(2, 0, 1)    # seq_len, batch, input_size
        output, (h_n, c_n) = self.lstm(x, None)
        output = output[-1:, :, :]
        output = output.view(output.size(1), -1)
        out = self.fc(output)
        return out
def polarity_lstm(**kwargs):
    r"""A LSTM based model.
    Kwargs (form like a dict and should be pass like **kwargs):
      hidden_size (default 64): recommended to be similar as the length of trimmed subsequence
      num_layers (default 2): layers are stacked and results are from the final layer
      start (default 250): start index of the subsequence
      end (default 350): end index of the subsequence
      bidirectional (default False): run lstm from left to right and from right to left
      contains_unkown (default False): True if targets have 0,1,2
    """
    default_kwargs = {"hidden_size":64, 
                      "num_layers":2,
                      "start": 250,
                      "end": 350,
                      "bidirectional":False, 
                      "contains_unkown":False}
    for k,v in kwargs.items():
      if k in default_kwargs:
        default_kwargs[k] = v
    print("#### model parameters ####\n")
    print(default_kwargs)
    print("\n##########################")
    if(default_kwargs['end'] < default_kwargs['start']):
      raise ValueError('<-- end cannot be smaller than start -->')
    model = PolarityLSTM(**default_kwargs)
    return model
