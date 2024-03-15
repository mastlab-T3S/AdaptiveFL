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
    def __init__(self, ResidualBlock, num_channels=3, num_classes=10, track_running_stats=True, scale=1.0, dataset='cifar',
                 exit1=6, exit2=7):
        super(ResNet, self).__init__()

        self.dataset = dataset

        self.exit1 = exit1
        self.exit2 = exit2

        self.inchannel = int(64 * scale)

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inchannel, track_running_stats=track_running_stats),
            nn.ReLU()
        )

        self.block = nn.Sequential()
        layer1 = self._make_layer(ResidualBlock, int(64 * scale), 2, stride=1,
                                  track_running_stats=track_running_stats)

        layer2 = self._make_layer(ResidualBlock, int(128 * scale), 2, stride=2,
                                  track_running_stats=track_running_stats)

        layer3 = self._make_layer(ResidualBlock, int(256 * scale), 2, stride=2,
                                  track_running_stats=track_running_stats)

        layer4 = self._make_layer(ResidualBlock, int(512 * scale), 2, stride=2,
                                  track_running_stats=track_running_stats)
        self.block.append(layer1[0])
        self.block.append(layer1[1])
        self.block.append(layer2[0])
        self.block.append(layer2[1])
        self.block.append(layer3[0])
        self.block.append(layer3[1])
        self.block.append(layer4[0])
        self.block.append(layer4[1])

        self.classifier = nn.ModuleList()
        # magic number matches the output size of the exit1 / exit2
        self.classifier.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Flatten(),
                                             nn.Linear(int(2 ** ((exit1 + 1) // 2) * 32 * scale), num_classes)))
        self.classifier.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Flatten(),
                                             nn.Linear(int(2 ** ((exit2 + 1) // 2) * 32 * scale), num_classes)))
        self.classifier.append(nn.Sequential(nn.AvgPool2d(4),
                                             nn.Flatten(),
                                             nn.Linear(int(512 * scale), num_classes)))

    def _make_layer(self, block, channels, num_blocks, stride, track_running_stats):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, track_running_stats))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x, ee=1):
        conv_out = self.conv1(x)
        if ee == 1:
            block1out = self.block[:self.exit1](conv_out)
            output = self.classifier[0](block1out)
            result = {'output': output}
            return [result]
        elif ee == 2:
            out = []
            block1out = self.block[:self.exit1](conv_out)
            output = self.classifier[0](block1out)
            result = {'output': output}
            out.append(result)

            block2out = self.block[self.exit1:self.exit2](block1out)
            output = self.classifier[1](block2out)
            result = {'output': output}
            out.append(result)

            return out
        elif ee == 3:
            out = []
            block1out = self.block[:self.exit1](conv_out)
            output = self.classifier[0](block1out)
            result = {'output': output}
            out.append(result)

            block2out = self.block[self.exit1:self.exit2](block1out)
            output = self.classifier[1](block2out)
            result = {'output': output}
            out.append(result)

            block3out = self.block[self.exit2:](block2out)
            output = self.classifier[2](block3out)
            result = {'output': output}
            out.append(result)

            return out


def ResNet18_cifar_scaleFL(num_channels=3, num_classes=10, track_running_stats=True, scale=1.0, exit1=6, exit2=7):
    return ResNet(ResidualBlock, num_channels, num_classes, track_running_stats, scale,
                  'cifar', exit1, exit2)  # 默认track_running_stats为true 即保留BN层的历史统计值


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet18_cifar_scaleFL(num_classes=10, track_running_stats=True, scale=1, exit1=6, exit2=7).to(device)
    print(model)
    total = 11173962
    for i in range(50, 100):
        scale = i / 100
        net = ResNet18_cifar_scaleFL(num_classes=10, track_running_stats=True, scale=scale, exit1=6, exit2=7).to(device)

        conv_params = sum(p.numel() for p in net.conv1.parameters())
        block_params = sum(p.numel() for p in net.block[:6].parameters())
        class_params = sum(p.numel() for p in net.classifier[2].parameters())

        print(f"scale for 0.25 length:{scale} width takes {(conv_params + block_params + class_params) / total}%")
