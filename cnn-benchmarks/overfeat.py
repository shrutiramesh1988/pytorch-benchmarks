import torch
import torch.nn as nn
#from .utils import load_state_dict_from_url


__all__ = ['Overfeat', 'overfeat']


#model_urls = {
#    'overfeat': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
#}


class Overfeat(nn.Module):

    def __init__(self, num_classes=1000):
        super(Overfeat, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024 * 6 * 6, 3072),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(3072, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def overfeat(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = Overfeat(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['overfeat'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
