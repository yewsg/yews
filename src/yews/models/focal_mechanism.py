import torch.nn as nn

# from .utils import load_state_dict_from_url

#import torch
#try:
#    from torch.hub import load_state_dict_from_url
#except ImportError:
#    from torch.utils.model_zoo import load_url as load_state_dict_from_url
    
__all__ = ['FmV1', 'fm_v1']

model_urls = {
    'fm_v1': 'https://www.dropbox.com/s/ckb4glf35agi9xa/fm_v1_wenchuan-bdd92da2.pth?dl=1',
}

class FmV1(nn.Module):

    def __init__(self):
        super().__init__()
        # n*3*71*9000
        # nn.Cov2d(3,16,kernel_size=(3,7),stride=1,padding=....)
        # (3,7) 3>71 7>18000 in the end use (3,3).

        # 71,9000 -> 71,4505
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3,13), stride=1, padding=(2,12), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2))
        )

        # 71,4505 -> 71,2259
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3,15), stride=1, padding=(2,14), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2))
        )

        # 71,2259 -> 71,1135
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,13), stride=1, padding=(2,12), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2))
        )

        # 71,1135 -> 71,572
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3,11), stride=1, padding=(2,10), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2))
        )

        # 71,572 -> 71,289
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3,9), stride=1, padding=(2,8), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2))
        )

        # 71,289 -> 71,143
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3,7), stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2))
        )

        # 71,143 -> 71,71
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3,5), stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2))
        )

        # 71,71 -> 35,35
        self.layer8 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 35,35 -> 17,17
        self.layer9 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 17,17 -> 8,8
        self.layer10 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        
        # 8,8 -> 4,4
        self.layer11 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 4,4 -> 2,2
        self.layer12 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 2,2 -> 1,1
        self.layer13 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Linear(64 * 1 * 1, 3)

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
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

def fm_v1(pretrained=False, progress=True, **kwargs):
    r"""Original CPIC model architecture from the
    `"Deep learning for ..." <https://arxiv.org/abs/1901.06396>`_ paper. The
    pretrained model is trained on 60,000 Wenchuan aftershock dataset
    demonstrated in the paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Wenchuan)
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = FmV1(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['fm_v1'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


#if __name__ == '__main__':
#    model = fm_v1(pretrained=False)
#    
#    x = torch.ones([1, 3, 71, 9000])
#    out = model(x)
#    print(out.size())
