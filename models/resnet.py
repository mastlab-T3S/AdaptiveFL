import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, track_running_stats=True):
        super(ResidualBlock, self).__init__()
        inchannel = inchannel
        outchannel = outchannel
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=track_running_stats)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:  # 这两个东西是在说一码事，再升维的时候需要增大stride来保持计算量
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel, track_running_stats=track_running_stats)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)  # 当shortcut是 nn.Sequential()的时候 会返回x本身
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_channels=3, num_classes=10, track_running_stats=True, slim_idx=0, scale=1.0, dataset='cifar'):
        super(ResNet, self).__init__()

        self.dataset = dataset
        if self.dataset == 'widar':
            self.reshape = nn.Sequential(
                nn.ConvTranspose2d(22, num_channels, 7, stride=1),
                nn.ReLU(),
                nn.ConvTranspose2d(num_channels, num_channels, kernel_size=7, stride=1),
                nn.ReLU()
            )

        idx = 0
        if idx < slim_idx:
            self.inchannel = 64
        else:
            self.inchannel = int(64 * scale)

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inchannel, track_running_stats=track_running_stats),
            nn.ReLU()
        )

        idx += 1
        tmp_scale = 1.0 if idx < slim_idx else scale
        self.layer1 = self._make_layer(ResidualBlock, int(96 * tmp_scale), 2, stride=1, track_running_stats=track_running_stats)

        idx += 1
        tmp_scale = 1.0 if idx < slim_idx else scale
        self.layer2 = self._make_layer(ResidualBlock, int(128 * tmp_scale), 2, stride=2, track_running_stats=track_running_stats)

        idx += 1
        tmp_scale = 1.0 if idx < slim_idx else scale
        self.layer3 = self._make_layer(ResidualBlock, int(256 * tmp_scale), 2, stride=2, track_running_stats=track_running_stats)

        idx += 1
        tmp_scale = 1.0 if idx < slim_idx else scale
        self.layer4 = self._make_layer(ResidualBlock, int(512 * tmp_scale), 2, stride=2, track_running_stats=track_running_stats)

        idx += 1
        tmp_scale = 1.0 if idx < slim_idx else scale
        self.fc = nn.Linear(int(512 * tmp_scale), num_classes)
        idx += 1

    def _make_layer(self, block, channels, num_blocks, stride, track_running_stats):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, track_running_stats))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.dataset == 'widar':
            out = self.conv1(self.reshape(x))
        else:
            out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        result = {'representation': out}
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        result['output'] = out
        return result


def ResNet18_cifar(num_channels=3, num_classes=10, track_running_stats=True, slim_idx=0, scale=1.0):
    return ResNet(ResidualBlock, num_channels, num_classes, track_running_stats, slim_idx, scale, 'cifar')  # 默认track_running_stats为true 即保留BN层的历史统计值


def ResNet18_widar(num_channels=3, num_classes=22, track_running_stats=True, slim_idx=0, scale=1.0):
    return ResNet(ResidualBlock, num_channels, num_classes, track_running_stats, slim_idx, scale, 'widar')


if __name__ == '__main__':
    net_1 = ResNet18_cifar(num_classes=10, track_running_stats=True, slim_idx=2, scale=0.45)
    #     net_2 = ResNet18(num_classes=10, track_running_stats=False, slim_idx=2, scale=0.75)
    print(net_1)
