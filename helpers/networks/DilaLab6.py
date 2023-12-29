import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from helpers.networks.share import *
from torchvision import transforms as T

class DilaLab6(nn.Module):
    def __init__(self, ch_in=1, ch_out=1):
        super(DilaLab6, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=ch_in, ch_out=16)
        self.Conv2 = conv_block(ch_in=16, ch_out=32)
        self.Conv3 = conv_block(ch_in=32, ch_out=64)
        self.Conv4 = conv_block(ch_in=64, ch_out=128)
        self.Conv5 = conv_block(ch_in=128, ch_out=256)
        self.Conv6 = conv_block(ch_in=256, ch_out=512)

        self.dilate = DB(ch_in=512, ch_out=512)  # cat

        self.Up6 = up_conv(ch_in=512, ch_out=512)
        self.Up_conv6 = conv_block(ch_in=1024, ch_out=512)

        self.Up5 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Up1 = up_conv(ch_in=32, ch_out=16)
        self.Up_conv1 = conv_block(ch_in=32, ch_out=16)

        self.Conv_1x1 = nn.Conv2d(16, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        x6 = self.Maxpool(x5)
        x6 = self.Conv6(x6)

        xp = self.Maxpool(x6)
        db6 = self.dilate(xp)

        d6 = self.Up6(db6)
        d6 = torch.cat((x6, d6), dim=1)
        d6 = self.Up_conv6(d6)

        d5 = self.Up5(d6)
        d5 = torch.cat((x5, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.Up_conv1(d1)

        d0 = self.Conv_1x1(d1)

        return d0


if __name__ == '__main__':
    import torch
    from thop import profile

    # Model
    print('==> Building model..')
    model = UNet()

    im = torch.rand((1, 1, 512, 512))
    flops, params = profile(model, (im,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
