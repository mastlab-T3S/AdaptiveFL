import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from torchinfo import summary


class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t, trs):
        super(LinearBottleNeck, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t, track_running_stats=trs),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t, track_running_stats=trs),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels, track_running_stats=trs)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class MobileNetV2(nn.Module):
    """
        MobileMetV2 implementation
    """

    def __init__(self, channels, num_classes, trs=True, slim_idx=0, scale=1.0):
        super(MobileNetV2, self).__init__()
        idx = 0
        tmp_scale = 1.0 if idx < slim_idx else scale
        self.pre = nn.Sequential(
            nn.Conv2d(channels, int(32 * tmp_scale), 3, padding=1),
            nn.BatchNorm2d(int(32 * tmp_scale), track_running_stats=trs),
            nn.ReLU6(inplace=True)
        )

        in_channels = int(32 * tmp_scale)
        idx += 1
        tmp_scale = 1.0 if idx < slim_idx else scale
        self.stage1 = LinearBottleNeck(in_channels, int(16 * tmp_scale), 1, 1, trs)

        in_channels = int(16 * tmp_scale)
        idx += 1
        tmp_scale = 1.0 if idx < slim_idx else scale
        self.stage2 = self._make_stage(2, in_channels, int(24 * tmp_scale), 2, 6, trs)

        in_channels = int(24 * tmp_scale)
        idx += 1
        tmp_scale = 1.0 if idx < slim_idx else scale
        self.stage3 = self._make_stage(3, in_channels, int(32 * tmp_scale), 2, 6, trs)

        in_channels = int(32 * tmp_scale)
        idx += 1
        tmp_scale = 1.0 if idx < slim_idx else scale
        self.stage4 = self._make_stage(4, in_channels, int(64 * tmp_scale), 2, 6, trs)

        in_channels = int(64 * tmp_scale)
        idx += 1
        tmp_scale = 1.0 if idx < slim_idx else scale
        self.stage5 = self._make_stage(3, in_channels, int(96 * tmp_scale), 1, 6, trs)

        in_channels = int(96 * tmp_scale)
        idx += 1
        tmp_scale = 1.0 if idx < slim_idx else scale
        self.stage6 = self._make_stage(3, in_channels, int(160 * tmp_scale), 2, 6, trs)

        in_channels = int(160 * tmp_scale)
        idx += 1
        tmp_scale = 1.0 if idx < slim_idx else scale
        self.stage7 = LinearBottleNeck(in_channels, int(320 * tmp_scale), 1, 6, trs)

        in_channels = int(320 * tmp_scale)
        idx += 1
        tmp_scale = 1.0 if idx < slim_idx else scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, int(1280 * tmp_scale), 1),
            nn.BatchNorm2d(int(1280 * tmp_scale), track_running_stats=trs),
            nn.ReLU6(inplace=True)
        )

        in_channels = int(1280 * tmp_scale)
        self.conv2 = nn.Conv2d(in_channels, num_classes, 1)

    def _make_stage(self, n, in_channels, out_channels, stride, t, trs):
        layers = [LinearBottleNeck(in_channels, out_channels, stride, t, trs)]

        while n - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t, trs))
            n -= 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_max_pool2d(x, 1)
        result = {'representation': x.view(x.size(0), -1)}
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        result['output'] = x
        return result


if __name__ == '__main__':
    net = MobileNetV2(22, 22, False, 0, 0.23)

    print(net)

    summary(net, (50, 22, 32, 32))

    dummy_input = torch.randn(1, 22, 32, 32).to('cuda')
    flops, params = profile(net, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))
