import torch
import torch.nn as nn

from .utils import load_state_dict_from_url

# try:
#     from torch.hub import load_state_dict_from_url
# except ImportError:
#     from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['PolarityCNN', 'polarity_cnn', 'PolarityLSTM', 'polarity_lstm', 'PolarityCNNLSTM', 'polarity_cnn_lstm']

model_urls = {
    'polarity_cnn': 'https://www.dr',
}

class PolarityCNN(nn.Module):
    
    #https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    
    def __init__(self):
        super(PolarityV2, self).__init__()
        self.features = nn.Sequential(

            # 300 -> 150

            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),


            # 150 -> 75

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),


            # 75 -> 37

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),


            # 37 -> 18
  
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),


            # 18 -> 9

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),


            # 9 -> 4

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4, 2),
        )
            
    def forward(self, x):
      
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

def polarity_cnn(pretrained=False, progress=True, **kwargs):
    model = PolarityCNN(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['polarity_cnn'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

# note: please use only 1 gpu to run LSTM, https://github.com/pytorch/pytorch/issues/21108
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

class PolarityCNNLSTM(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        # 300 -> 150
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 150 -> 75
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        input_size = 64
        self.hidden_size = kwargs["hidden_size"]
        self.bidirectional = kwargs["bidirectional"]
        self.contains_unkown = kwargs["contains_unkown"]
        self.start = kwargs['start']
        self.end = kwargs['end']
        self.num_layers = kwargs['num_layers']
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), 3 if self.contains_unkown else 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.narrow(2,self.start, self.end-self.start)
        out = out.permute(2, 0, 1)    # seq_len, batch, input_size
        output, (h_n, c_n) = self.lstm(out, None)
        output = output[-1:, :, :]
        output = output.view(output.size(1), -1)
        out = self.fc(output)
        return out

def polarity_cnn_lstm(**kwargs):
    r"""A LSTM based model.
    Kwargs (form like a dict and should be pass like **kwargs):
      hidden_size (default 64): recommended to be similar as the length of trimmed subsequence
      num_layers (default 2): layers are stacked and results are from the final layer
      start (default 0): start index of the subsequence
      end (default 75): end index of the subsequence
      bidirectional (default False): run lstm from left to right and from right to left
      contains_unkown (default False): True if targets have 0,1,2
    """
    default_kwargs = {"hidden_size":64,
                      "num_layers":2,
                      "start": 0,
                      "end": 75,
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
    model = PolarityCNNLSTM(**default_kwargs)
    return model

# if __name__ == '__main__':
#     model = polarity_cnn(pretrained=False)
#     x = torch.ones([1, 1, 600])
#     out = model(x)
#     print(out.size())
