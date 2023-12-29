# *.* coding:utf-8 *.*
import torch.nn as nn
import torch
from torch.nn import init
import torch.nn.functional as F

affine_par = True


def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block_first(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_first, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out // 2), nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_out // 2, ch_out // 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out // 2), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = torch.cat((x0, x1), dim=1)
        x3 = self.conv2(x2)
        return x2, x3


class conv_block_cunet(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_cunet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_in), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.cat((x, x1), dim=1)
        x3 = self.conv2(x2)
        return x2, x3


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            # nn.ReLU(inplace=True),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            # nn.ReLU(inplace=True)
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(down_conv_block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=2, padding=0, bias=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return x1 + x2


# SE
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# AG
class Attention_block(nn.Module):
    def __init__(self, ch_g, ch_xl, ch_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(ch_g,
                      ch_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(ch_int))

        self.W_x = nn.Sequential(
            nn.Conv2d(ch_xl,
                      ch_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(ch_int))

        self.psi = nn.Sequential(
            nn.Conv2d(ch_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

#  SegNet

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()

        batchNorm_momentum = 0.1

        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),
        )

        self.encode2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),
        )

        self.encode3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),
        )

        self.encode4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),
        )

        self.encode5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        idx = []

        x = self.encode1(x)  # [1 64 512 512]
        x, id1 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)  # [1 64 256 256]  id1 = [1 64 256 256]
        idx.append(id1)

        x = self.encode2(x)  # [1 128 256 256]
        x, id2 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)  # [1 128 128 128] id2 = [1 128 128 128]
        idx.append(id2)

        x = self.encode3(x)  # [1 256 128 128]
        x, id3 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)   # [1 256 64 64]  id3 = [1 256 64 64]
        idx.append(id3)

        x = self.encode4(x)  # [1 512 64 64]
        x, id4 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)   # [1 512 32 32]  id4 = [1 512 32 32]
        idx.append(id4)

        x = self.encode5(x)  # [1 512 32 32]
        x, id5 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)  # [1 512 16 16]  id5 = [1 512 16 16]
        idx.append(id5)

        return x, idx

#  SegNet
class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()

        batchNorm_momentum = 0.1

        self.decode1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True)
        )

        self.decode2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True)
        )

        self.decode3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True)
        )

        self.decode4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True)
        )

        self.decode5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, idx):
        x = F.max_unpool2d(x, idx[4], kernel_size=2, stride=2)  # [1 512 32 32]

        x = self.decode1(x)  # [1 512 32 32]
        x = F.max_unpool2d(x, idx[3], kernel_size=2, stride=2)  # [1 512 64 64]

        x = self.decode2(x)  # [1 256 64 64]
        x = F.max_unpool2d(x, idx[2], kernel_size=2, stride=2)  # [1 256 128 128]

        x = self.decode3(x) # [1 128 128 128]
        x = F.max_unpool2d(x, idx[1], kernel_size=2, stride=2) # [1 128 256 256]

        x = self.decode4(x)  # [1 64 256 256]
        x = F.max_unpool2d(x, idx[0], kernel_size=2, stride=2)   # [1 64 512 512]

        x = self.decode5(x)

        return x

# DilaLab6
class DB(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(DB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, dilation=1, padding=1),
            torch.nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, dilation=2, padding=2),
            torch.nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, dilation=4, padding=4),
            torch.nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, dilation=8, padding=8),
            torch.nn.ReLU())

    def forward(self, x):
        conv1_output = self.conv1(x)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)

        output = x + conv1_output + conv2_output + conv3_output + conv4_output
        return output

# Non-Local
class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class Attention_block(nn.Module):
    def __init__(self, ch_g, ch_xl, ch_int):
        F_g = ch_g
        F_l = ch_xl
        F_int = ch_int
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class conv_block_dilation(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_dilation, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# class RCCAModule(nn.Module):
#     def __init__(self, ch_in, ch_out, num_classes):
#         super(RCCAModule, self).__init__()
#         inter_channels = ch_in // 4
#         self.conva = nn.Sequential(nn.Conv2d(ch_in, inter_channels, 3, padding=1, bias=False),
#                                    nn.BatchNorm2d(inter_channels),nn.ReLU(inplace=False))
#         self.cca = CrissCrossAttention(inter_channels)
#         self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
#                                    nn.BatchNorm2d(inter_channels),nn.ReLU(inplace=False))
#
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(ch_in+inter_channels, ch_out, kernel_size=3, padding=1, dilation=1, bias=False),
#             nn.BatchNorm2d(ch_out),nn.ReLU(inplace=False),
#             nn.Dropout2d(0.1),
#             #nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
#             )
#     def forward(self, x, recurrence=2):
#         output = self.conva(x)
#         for i in range(recurrence):
#             output = self.cca(output)
#         output = self.convb(output)
#
#         output = self.bottleneck(torch.cat([x, output], 1))
#         return output


class multi_head_attention_2d(nn.Module):
    def __init__(self, in_channel, key_filters, value_filters, output_filters, num_heads, dropout_prob=0.5,
                 layer_type='SAME'):
        super().__init__()
        """Multihead scaled-dot-product attention with input/output transformations.

        Args:
            inputs: a Tensor with shape [batch, h, w, channels]
            key_filters: an integer. Note that queries have the same number 
                of channels as keys
            value_filters: an integer
            output_depth: an integer
            num_heads: an integer dividing key_filters and value_filters
            layer_type: a string, type of this layer -- SAME, DOWN, UP
        Returns:
            A Tensor of shape [batch, _h, _w, output_filters]

        Raises:
            ValueError: if the key_filters or value_filters are not divisible
                by the number of attention heads.
        """

        if key_filters % num_heads != 0:
            raise ValueError("Key depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (key_filters, num_heads))
        if value_filters % num_heads != 0:
            raise ValueError("Value depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (value_filters, num_heads))
        if layer_type not in ['SAME', 'DOWN', 'UP']:
            raise ValueError("Layer type (%s) must be one of SAME, "
                             "DOWN, UP." % (layer_type))

        self.num_heads = num_heads
        self.layer_type = layer_type

        self.QueryTransform = None
        if layer_type == 'SAME':
            self.QueryTransform = nn.Conv2d(in_channel, key_filters, kernel_size=1, stride=1,
                                            padding=0, bias=True)
        elif layer_type == 'DOWN':
            self.QueryTransform = nn.Conv2d(in_channel, key_filters, kernel_size=3, stride=2,
                                            padding=1, bias=True)  # author use bias
        elif layer_type == 'UP':
            self.QueryTransform = nn.ConvTranspose2d(in_channel, key_filters, kernel_size=3, stride=2,
                                                     padding=1, bias=True)

        self.KeyTransform = nn.Conv2d(in_channel, key_filters, kernel_size=1, stride=1, padding=0, bias=True)
        self.ValueTransform = nn.Conv2d(in_channel, value_filters, kernel_size=1, stride=1, padding=0, bias=True)
        self.attention_dropout = nn.Dropout(dropout_prob)

        self.outputConv = nn.Conv2d(value_filters, output_filters, kernel_size=1, stride=1, padding=0, bias=True)

        self._scale = (key_filters // num_heads) ** 0.5

    def forward(self, inputs):
        """
        :param inputs: B, C, H, W
        :return: inputs: B, Co, Hq, Wq
        """

        if self.layer_type == 'SAME' or self.layer_type == 'DOWN':
            q = self.QueryTransform(inputs)
        elif self.layer_type == 'UP':
            q = self.QueryTransform(inputs, output_size=(inputs.shape[2] * 2, inputs.shape[3] * 2))

        # [B, Hq, Wq, Ck]
        k = self.KeyTransform(inputs).permute(0, 2, 3, 1)
        v = self.ValueTransform(inputs).permute(0, 2, 3, 1)
        q = q.permute(0, 2, 3, 1)

        Batch, Hq, Wq = q.shape[0], q.shape[1], q.shape[2]

        # [B, H, W, N, Ck]
        k = self.split_heads(k, self.num_heads)
        v = self.split_heads(v, self.num_heads)
        q = self.split_heads(q, self.num_heads)

        # [(B, H, W, N), c]
        k = torch.flatten(k, 0, 3)
        v = torch.flatten(v, 0, 3)
        q = torch.flatten(q, 0, 3)

        # normalize
        q = q / self._scale
        # attention
        # [(B, Hq, Wq, N), (B, H, W, N)]
        A = torch.matmul(q, k.transpose(0, 1))
        A = torch.softmax(A, dim=1)
        A = self.attention_dropout(A)

        # [(B, Hq, Wq, N), C]
        O = torch.matmul(A, v)
        # [B, Hq, Wq, C]
        O = O.view(Batch, Hq, Wq, v.shape[-1] * self.num_heads)
        # [B, C, Hq, Wq]
        O = O.permute(0, 3, 1, 2)
        # [B, Co, Hq, Wq]
        O = self.outputConv(O)

        return O

    def split_heads(self, x, num_heads):
        """Split channels (last dimension) into multiple heads.

        Args:
            x: a Tensor with shape [batch, h, w, channels]
            num_heads: an integer

        Returns:
            a Tensor with shape [batch, h, w, num_heads, channels / num_heads]
        """

        channel_num = x.shape[-1]
        return x.view(x.shape[0], x.shape[1], x.shape[2], num_heads, int(channel_num / num_heads))
