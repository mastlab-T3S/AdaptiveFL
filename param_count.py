import torch
from torchinfo import summary

from models import MobileNetV2
from models.resnet import ResNet18_cifar
from models.vgg import vgg_16_bn
from utils.options import args_parser
from thop import profile

args = args_parser()

def show_vgg_param():
    for slim_idx in range(1, 10):
        for ration in [65]:
            net = vgg_16_bn(num_classes=10, track_running_stats=False, num_channels=args.num_channels, slim_idx=slim_idx, scale=ration / 100 * 4)
            total = sum([param.nelement() for param in net.parameters()])
            print(f'slim_idx: {slim_idx}  ration: {ration} , Number of parameter: % .4fM' % (total / 1e6))
        print()


def show_resnet_param():
    for slim_idx in range(6):
        for ration in [0.43,0.68]:
            net = ResNet18_cifar(num_classes=args.num_classes, track_running_stats=False, slim_idx=slim_idx, scale=ration)
            total = sum([param.nelement() for param in net.parameters()])
            print(f'slim_idx: {slim_idx}  ration: {ration} , Number of parameter: % .4fM' % (total / 1e6))
        print()


def show_mobilenet_param():
    for slim_idx in range(2,5):
        for ration in [0.46,0.69,1.0]:
            net = MobileNetV2(num_classes=args.num_classes, trs=False, slim_idx=slim_idx, scale=ration)
            total = sum([param.nelement() for param in net.parameters()])
            print(f'slim_idx: {slim_idx}  ration: {ration} , Number of parameter: % .4fM' % (total / 1e6))
        print()


if __name__ == '__main__':
    # show_vgg_param()
    # show_resnet_param()
    show_mobilenet_param()
    # net = ResNet18_cifar(num_classes=10, track_running_stats=False, slim_idx=0, scale=4).to('cuda')
    net = vgg_16_bn(num_classes=10, track_running_stats=False, slim_idx=0, scale=1.0).to('cuda')

    summary(net, (50, 3, 32, 32))
    #
    # dummy_input = torch.randn(1, 3, 32, 32).to('cuda')
    # flops, params = profile(net, (dummy_input,))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params /1e6))
