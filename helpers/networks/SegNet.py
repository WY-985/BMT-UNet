import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from helpers.networks.share import *
from torchvision import transforms as T

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()

        self.encode = Encoder(in_channels=1)
        self.decode = Decoder(out_channels=1)

    def forward(self, x):
        x, idx = self.encode(x)  # 1 512 16 16  idx[5]
        x = self.decode(x, idx)  # 1 1 512 512
        return x


if __name__ == '__main__':

    import torch
    from thop import profile

    # Model
    print('==> Building model..')
    model = SegNet()

    im = torch.rand((1, 1, 512, 512))
    flops, params = profile(model, (im,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))


