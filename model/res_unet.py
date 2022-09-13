# -*- encoding: utf-8 -*-
# @File    :   ResUnet.py
# @Time    :   2021/05/28 00:18:27
# @Author  :   Gorkem Can Ates
# @Contact :   g.canates@gmail.com
# @Desc    :   None

from berries.model.base import BaseModel
import torch
import torch.nn as nn
from model.main_block import conv_block, ResConv


class ResUnet(BaseModel):
    def __init__(self, in_features=3, out_features=3, k=1, norm_type='bn'):
        nn.Module.__init__(self)
        self.encode1 = nn.Sequential(
            conv_block(in_features=in_features,
                       out_features=int(64 * k),
                       norm_type=norm_type),
            nn.Conv2d(in_channels=int(64 * k),
                      out_channels=int(64 * k),
                      kernel_size=3,
                      padding=1))
        self.skip = nn.Conv2d(in_channels=in_features,
                              out_channels=int(64 * k),
                              kernel_size=3,
                              padding=1)
        self.encode2 = ResConv(in_features=int(64 * k),
                               out_features=int(128 * k),
                               stride=2,
                               norm_type=norm_type)
        self.encode3 = ResConv(in_features=int(128 * k),
                               out_features=int(256 * k),
                               stride=2,
                               norm_type=norm_type)
        self.latent = ResConv(in_features=int(256 * k),
                              out_features=int(512 * k),
                              stride=2,
                              norm_type=norm_type)
        self.upsample1 = nn.ConvTranspose2d(in_channels=int(512 * k),
                                            out_channels=int(512 * k),
                                            kernel_size=2,
                                            stride=2)
        self.decode1 = ResConv(in_features=(int(512 * k) + int(256 * k)),
                               out_features=int(256 * k),
                               norm_type=norm_type)
        self.upsample2 = nn.ConvTranspose2d(in_channels=int(256 * k),
                                            out_channels=int(256 * k),
                                            kernel_size=2,
                                            stride=2)
        self.decode2 = ResConv(in_features=(int(256 * k) + int(128 * k)),
                               out_features=int(128 * k),
                               norm_type=norm_type)
        self.upsample3 = nn.ConvTranspose2d(in_channels=int(128 * k),
                                            out_channels=int(128 * k),
                                            kernel_size=2,
                                            stride=2)
        self.decode3 = ResConv(in_features=(int(128 * k) + int(64 * k)),
                               out_features=int(64 * k),
                               norm_type=norm_type)
        # self.upsample4 = nn.ConvTranspose2d(in_channels=int(64 * k), out_channels=int(64 * k),
        #                                     kernel_size=2, stride=2)
        # self.decode4 = ResConv(in_features=(int(64 * k)), out_features=int(64 * k),
        #                        norm_type=norm_type)
        self.output = nn.Conv2d(in_channels=int(64 * k),
                                out_channels=out_features,
                                kernel_size=1,
                                padding=0)

    def forward(self, x):
        x1 = self.encode1(x) + self.skip(x)
        x2 = self.encode2(x1)
        x3 = self.encode3(x2)
        x = self.latent(x3)
        x = self.upsample1(x)
        x = self.decode1(torch.cat((x, x3), dim=1))
        x = self.upsample2(x)
        x = self.decode2(torch.cat((x, x2), dim=1))
        x = self.upsample3(x)
        x = self.decode3(torch.cat((x, x1), dim=1))
        # x = self.upsample4(x)
        # x = self.decode4(x)
        x = self.output(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':

    def test(batchsize):
        in_channels = 3
        in1 = torch.rand(batchsize, in_channels, 512, 512).to('cuda')
        model = ResUnet(in_features=in_channels,
                        out_features=3,
                        k=0.25,
                        norm_type='gn').to('cuda')

        out1 = model(in1)
        total_params = sum(p.numel() for p in model.parameters())

        return out1.shape, total_params

    shape, total_params = test(batchsize=6)
    print('Shape : ', shape, '\nTotal params : ', total_params)
