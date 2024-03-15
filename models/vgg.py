"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
import torch
from thop import profile
from torchinfo import summary

'''VGG11/13/16/19 in Pytorch.'''

import torch.nn as nn

from utils.scaler import Scaler

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}




class VGG(nn.Module):
    def __init__(self, features, num_class=100, num_channels=3, scale=1.0):
        super().__init__()
        self.features = features

        if num_channels == 3:
            dim = 4096
        else:
            dim = 256

        self.projector = nn.Sequential(
            nn.Linear(int(512 * scale), int(dim * scale)),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(int(dim * scale), int(dim * scale)),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.classifier = nn.Linear(int(dim * scale), num_class)

    def forward(self, x):

        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.projector(output)
        result = {'representation': output}
        output = self.classifier(output)
        result['output'] = output
        return result


def make_layers(cfg, batch_norm=False, track_running_stats=True, num_channels=3, slim_idx=2, scale=1.0):
    layers = []
    num = 0
    input_channel = num_channels

    if num_channels == 3:
        cfg.append('M')
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        if num >= slim_idx:
            l = int(l * scale)

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
        if batch_norm:
            layers += [nn.BatchNorm2d(l, track_running_stats=track_running_stats)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l
        num += 1

    if num_channels == 3:
        cfg.pop()
    return nn.Sequential(*layers)


def vgg_16_bn(num_classes, track_running_stats=True, num_channels=3, slim_idx=0, scale=1.0):
    return VGG(make_layers(cfg['D'], batch_norm=True, track_running_stats=track_running_stats, num_channels=num_channels, slim_idx=slim_idx, scale=scale),
               num_class=num_classes,
               num_channels=num_channels, scale=scale)


if __name__ == '__main__':
    net = vgg_16_bn(10, False, 3, 8, 0.40)

    summary(net, (50, 3, 32, 32))

    dummy_input = torch.randn(1, 3, 32, 32).to('cuda')
    flops, params = profile(net, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))
