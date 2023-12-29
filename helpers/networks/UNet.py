import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from helpers.networks.share import *
from torchvision import transforms as T


class UNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1)
        self.Conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.Conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        self.Conv_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        self.Conv_5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        # dropout可以作为训练神经网络的一种trick选择,在每个训练批次中,以某种概率忽略一定数量的神经元.可以明显地减少过拟合现象.
        x1 = self.Conv1(x)
        x1 = F.dropout(x1, p=0.2)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2 = F.dropout(x2, p=0.2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = F.dropout(x3, p=0.2)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4 = F.dropout(x4, p=0.2)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5 = F.dropout(x5, p=0.2)

        d5 = self.Up5(x5)  # 512
        d5 = torch.cat((x4, d5), dim=1)  # 1024
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        return d1


class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()

        self.encode = Encoder(in_channels=1)
        self.decode = Decoder(out_channels=1)

    def forward(self, x):
        x, idx = self.encode(x)
        x = self.decode(x, idx)
        return x


if __name__ == '__main__':
    # im = Image.open('85.png')
    # # T?
    # aug = T.Compose([T.Resize(512),
    #                  T.ToTensor()])
    # im = aug(im)
    # im = im.unsqueeze(0)
    # model = UNet(1, 1)
    # sr = torch.sigmoid(model(im))
    #
    # #可视化
    # sr = sr.detach().numpy()
    # sr = sr.squeeze(0)
    # sr = sr.squeeze(0)
    # plt.imshow(sr, cmap='gray')
    # plt.show()
    #
    # #在下面这打断点
    # print('finish')
    # -- coding: utf-8 --
    import torch
    from thop import profile

    # Model
    print('==> Building model..')
    model = UNet()

    im = torch.rand((1, 1, 512, 512))
    flops, params = profile(model, (im,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
    # -- coding: utf-8 --
    # import torchvision
    # from ptflops import get_model_complexity_info
    #
    # # model = torchvision.models.alexnet(pretrained=False)
    # model = UNet()
    # flops, params = get_model_complexity_info(model, (1, 512, 512), as_strings=True, print_per_layer_stat=True)
    # print('flops: ', flops, 'params: ', params)

    # from ptflops import get_model_complexity_info
    # from torchvision.models import resnet18
    # model = resnet18()
    # flops, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True, print_per_layer_stat=True)
    # print('Flops:  ' + flops)
    # print('Params: ' + params)
