__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"


import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


######### XNET #########
class conv_block(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, stride=1, padding=1, dilation=1, norm_type=None,
                 use_bias=True):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=use_bias)

        self.norm_type = norm_type
        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        x = self.relu(x)
        return x


class DepthWiseConv2D(nn.Module):
    def __init__(self, in_features, kernels_per_layer=1):
        nn.Module.__init__(self)
        self.depthwise = nn.Conv2d(
            in_features, in_features * kernels_per_layer, kernel_size=3, padding=1, groups=in_features, dilation=1)
        self.pointwise = nn.Conv2d(in_features * kernels_per_layer, in_features, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DSconv_block(nn.Module):
    def __init__(self, in_features, out_features, norm_type):
        nn.Module.__init__(self)
        self.norm_type = norm_type
        self.DSconv = DepthWiseConv2D(in_features)
        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if (in_features >= 32 and in_features % 32 == 0) else in_features, in_features)

        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(in_features)
        self.relu = nn.ReLU()
        self.conv2d = conv_block(in_features, out_features, kernel_size=1, padding=0, norm_type=norm_type)

    def forward(self, x):
        x = self.DSconv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        x = self.relu(x)
        x = self.conv2d(x)
        return x


class x_block(nn.Module):
    def __init__(self, in_features, out_features, norm_type):
        nn.Module.__init__(self)
        self.res = conv_block(in_features, out_features, kernel_size=1, padding=0, norm_type=norm_type)
        self.dc1 = DSconv_block(in_features, out_features, norm_type=norm_type)
        self.dc2 = DSconv_block(out_features, out_features, norm_type=norm_type)

    def forward(self, x):
        res = self.res(x)
        x = self.dc1(x)
        x = self.dc2(x)
        x = self.dc2(x)
        x += res
        return x


class FSM(nn.Module):
    def __init__(self, norm_type, device):
        nn.Module.__init__(self)
        self.norm_type = norm_type
        self.device = device

    def forward(self, x):
        channel_num = x.shape[1]
        res = x
        x = conv_block(
            in_features=int(channel_num), out_features=int(channel_num // 8), norm_type=self.norm_type).to(self.device)(
            x)

        ip = x
        batchsize, channels, dim1, dim2 = ip.shape
        intermediate_dim = int(channels // 2)

        theta = nn.Conv2d(int(channel_num // 8), intermediate_dim, kernel_size=1, padding=0, bias=False).to(
            self.device)(ip)
        theta = torch.reshape(theta, (batchsize, -1, intermediate_dim))

        phi = nn.Conv2d(int(channel_num // 8), intermediate_dim, kernel_size=1, padding=0, bias=False).to(self.device)(
            ip)
        phi = torch.reshape(phi, (batchsize, -1, intermediate_dim))

        f = torch.bmm(theta, phi.view(batchsize, intermediate_dim, phi.shape[1]))
        f = f / (float(f.shape[-1]))

        g = nn.Conv2d(int(channel_num // 8), intermediate_dim, kernel_size=1, padding=0, bias=False).to(self.device)(ip)
        g = torch.reshape(g, (batchsize, -1, intermediate_dim))

        y = torch.bmm(f, g)
        y = torch.reshape(y, (batchsize, intermediate_dim, dim1, dim2))
        y = nn.Conv2d(intermediate_dim, channels, kernel_size=1, padding=0, bias=False).to(self.device)(y)
        y += ip
        x = y
        x = conv_block(in_features=channels, out_features=int(channel_num), norm_type=self.norm_type).to(self.device)(x)
        x += res

        return x


######### ResUnet & ResUnetPlus #########


class ResConv(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=1, norm_type=None):
        nn.Module.__init__(self)
        self.norm_type = norm_type
        if self.norm_type == 'gn':
            self.norm1 = nn.GroupNorm(32 if (in_features >= 32 and in_features % 32 == 0) else in_features, in_features)
            self.norm2 = nn.GroupNorm(32 if (out_features >= 32 and out_features % 32 == 0) else out_features,
                                      out_features)

        if self.norm_type == 'bn':
            self.norm1 = nn.BatchNorm2d(in_features)
            self.norm2 = nn.BatchNorm2d(out_features)

        self.pack = nn.Sequential(self.norm1,
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=in_features, out_channels=out_features,
                                            kernel_size=kernel_size, stride=stride, padding=padding),
                                  self.norm2,
                                  nn.ReLU(),

                                  nn.Conv2d(in_channels=out_features, out_channels=out_features,
                                            kernel_size=kernel_size, stride=1, padding=1))
        self.skip = nn.Sequential(nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                                            stride=stride, padding=1),
                                  self.norm2)

    def forward(self, x):
        res = self.skip(x)
        x = self.pack(x)
        x += res
        return x


class SqueezeExciteBlock(nn.Module):
    def __init__(self, in_features, reduction=16, squeeze_flag=False):
        nn.Module.__init__(self)
        self.squeeze_flag = squeeze_flag
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if self.squeeze_flag:
            self.squeeze = Squeeze(in_features, norm_type='gn')
        self.fc = nn.Sequential(nn.Linear(in_features, int(in_features // reduction), bias=False),
                                nn.ReLU(),
                                nn.Linear(int(in_features // reduction), in_features, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        x_init = x
        b, c, _, _ = x.size()
        if self.squeeze_flag:
            x = self.squeeze(x)
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x_init * y.expand_as(x_init)


class Squeeze(nn.Module):
    def __init__(self, in_features, norm_type=None):
        super(Squeeze, self).__init__()

        self.squeeze1 = conv_block(in_features, in_features, kernel_size=9, stride=8, padding=4, norm_type=norm_type)
        self.squeeze2 = conv_block(in_features, in_features, kernel_size=7, stride=4, padding=3, norm_type=norm_type)
        self.squeeze3 = conv_block(in_features, in_features, kernel_size=5, stride=4, padding=2, norm_type=norm_type)

    def forward(self, x):
        x1 = self.squeeze1(x)
        x2 = self.squeeze2(x1)
        x3 = self.squeeze3(x2)
        return x3


class ResUASPP(nn.Module):
    def __init__(self, in_features, out_features, norm_type, rate=[6, 12, 18]):
        nn.Module.__init__(self)

        self.block1 = conv_block(
            in_features=in_features,
            out_features=out_features,
            padding=rate[0],
            dilation=rate[0],
            norm_type=norm_type)
        self.block2 = conv_block(
            in_features=in_features,
            out_features=out_features,
            padding=rate[1],
            dilation=rate[1],
            norm_type=norm_type)
        self.block3 = conv_block(
            in_features=in_features,
            out_features=out_features,
            padding=rate[2],
            dilation=rate[2],
            norm_type=norm_type)

        self.out = nn.Conv2d(
            in_channels=int(len(rate) * out_features),
            out_channels=out_features,
            kernel_size=(1, 1))

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x = self.out(torch.cat((x1, x2, x3), dim=1))
        return x


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim, norm_type):
        nn.Module.__init__(self)
        if norm_type == 'gn':
            self.norm1 = nn.GroupNorm(32 if (input_encoder >= 32 and input_encoder % 32 == 0) else input_encoder,
                                      input_encoder)
            self.norm2 = nn.GroupNorm(32 if (input_decoder >= 32 and input_decoder % 32 == 0) else input_decoder,
                                      input_decoder)
            self.norm3 = nn.GroupNorm(32 if (output_dim >= 32 and output_dim % 32 == 0) else output_dim,
                                      output_dim)

        if norm_type == 'bn':
            self.norm1 = nn.BatchNorm2d(input_encoder)
            self.norm2 = nn.BatchNorm2d(input_decoder)
            self.norm3 = nn.BatchNorm2d(output_dim)

        self.conv_encoder = nn.Sequential(
            self.norm1,
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            self.norm2,
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            self.norm3,
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2


############### DoubleUnet ###############

class DoubleASPP(nn.Module):
    def __init__(self, in_features, out_features, norm_type, rate=[1, 6, 12, 18], device='cuda'):
        nn.Module.__init__(self)
        self.device = device
        self.block0 = conv_block(
            in_features=in_features,
            out_features=out_features,
            kernel_size=1,
            padding=0,
            dilation=1,
            norm_type=norm_type,
            use_bias=False)
        self.block1 = conv_block(
            in_features=in_features,
            out_features=out_features,
            kernel_size=1,
            padding=0,
            dilation=rate[0],
            norm_type=norm_type,
            use_bias=False)
        self.block2 = conv_block(
            in_features=in_features,
            out_features=out_features,
            padding=rate[1],
            dilation=rate[1],
            norm_type=norm_type,
            use_bias=False)
        self.block3 = conv_block(
            in_features=in_features,
            out_features=out_features,
            padding=rate[1],
            dilation=rate[1],
            norm_type=norm_type,
            use_bias=False)
        self.block4 = conv_block(
            in_features=in_features,
            out_features=out_features,
            padding=rate[1],
            dilation=rate[1],
            norm_type=norm_type,
            use_bias=False)

        self.out = conv_block(
            in_features=int((len(rate) + 1) * out_features),
            out_features=out_features,
            kernel_size=1,
            padding=0,
            use_bias=False)

    def forward(self, x):
        x1 = nn.AvgPool2d(kernel_size=(x.shape[2], x.shape[3])).to(self.device)(x)
        x1 = self.block0(x1)
        x1 = nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear').to(self.device)(x1)
        x2 = self.block1(x)
        x3 = self.block2(x)
        x4 = self.block3(x)
        x5 = self.block4(x)
        x6 = self.out(torch.cat((x1, x2, x3, x4, x5), dim=1))
        return x6


############### R2AttNet ###############


class R2_Attention_block(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim, norm_type):
        nn.Module.__init__(self)
        if norm_type == 'gn':
            self.norm1 = nn.GroupNorm(32 if (output_dim >= 32 and output_dim % 32 == 0) else output_dim,
                                      output_dim)
            self.norm2 = nn.GroupNorm(32 if (output_dim >= 32 and output_dim % 32 == 0) else output_dim,
                                      output_dim)
            self.norm3 = nn.GroupNorm(1, 1)

        if norm_type == 'bn':
            self.norm1 = nn.BatchNorm2d(output_dim)
            self.norm2 = nn.BatchNorm2d(output_dim)
            self.norm3 = nn.BatchNorm2d(1)

        self.W_g = nn.Sequential(
            nn.Conv2d(input_encoder, output_dim, kernel_size=1, stride=1, padding=0, bias=True),
            self.norm1
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(input_decoder, output_dim, kernel_size=1, stride=1, padding=0, bias=True),
            self.norm2
        )

        self.psi = nn.Sequential(
            nn.Conv2d(output_dim, 1, kernel_size=1, stride=1, padding=0, bias=True),
            self.norm3,
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, norm_type):
        nn.Module.__init__(self)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv_block(in_features=ch_in, out_features=ch_out, norm_type=norm_type)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, norm_type, t=2):
        nn.Module.__init__(self)
        self.t = t
        self.ch_out = ch_out
        self.conv = conv_block(in_features=ch_out, out_features=ch_out, norm_type=norm_type)

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, norm_type, t=2):
        nn.Module.__init__(self)
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t, norm_type=norm_type),
            Recurrent_block(ch_out, t=t, norm_type=norm_type)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1
