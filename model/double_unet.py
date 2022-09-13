# -*- encoding: utf-8 -*-
# @File    :   ResUnet.py
# @Time    :   2021/05/29 19:46:27
# @Author  :   Gorkem Can Ates
# @Contact :   g.canates@gmail.com
# @Desc    :   None

from berries.model.base import BaseModel
import torch
import torch.nn as nn
import torchvision
from torchvision.models import vgg19
from model.main_block import conv_block, DoubleASPP, SqueezeExciteBlock


class DoubleUnet(BaseModel):
    def __init__(self,
                 out_features=3,
                 k=1,
                 k1=1,
                 norm_type='bn',
                 upsample_mode='bilinear',
                 device='cuda'):
        nn.Module.__init__(self)
        self.mu = torch.tensor([0.485, 0.456, 0.406],
                               requires_grad=False).to(device).view(
                                   (1, 3, 1, 1))
        self.sigma = torch.tensor([0.229, 0.224, 0.225],
                                  requires_grad=False).to(device).view(
                                      (1, 3, 1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)

        for params in vgg19(pretrained=True).features.parameters():
            params.requires_grad = True

        self.vgg1 = vgg19(pretrained=True).features[:4]
        self.vgg2 = vgg19(pretrained=True).features[4:9]
        self.vgg3 = vgg19(pretrained=True).features[9:18]
        self.vgg4 = vgg19(pretrained=True).features[18:27]
        self.vgg5 = vgg19(pretrained=True).features[27:-1]

        self.ASPP_1 = DoubleASPP(in_features=512,
                                 out_features=int(64 * k1),
                                 norm_type=norm_type)

        self.decode1_0 = self.upsample
        self.decode1_1 = nn.Sequential(
            conv_block(in_features=int(64 * k1) + 512,
                       out_features=int(256 * k),
                       norm_type=norm_type),
            conv_block(in_features=int(256 * k),
                       out_features=int(256 * k),
                       norm_type=norm_type),
            SqueezeExciteBlock(in_features=int(256 * k), reduction=8),
            self.upsample)
        self.decode1_2 = nn.Sequential(
            conv_block(in_features=int(256 * k) + 256,
                       out_features=int(128 * k),
                       norm_type=norm_type),
            conv_block(in_features=int(128 * k),
                       out_features=int(128 * k),
                       norm_type=norm_type),
            SqueezeExciteBlock(in_features=int(128 * k), reduction=8),
            self.upsample)
        self.decode1_3 = nn.Sequential(
            conv_block(in_features=int(128 * k) + 128,
                       out_features=int(64 * k),
                       norm_type=norm_type),
            conv_block(in_features=int(64 * k),
                       out_features=int(64 * k),
                       norm_type=norm_type),
            SqueezeExciteBlock(in_features=int(64 * k), reduction=8),
            self.upsample)
        self.decode1_4 = nn.Sequential(
            conv_block(in_features=int(64 * k) + 64,
                       out_features=int(32 * k),
                       norm_type=norm_type),
            conv_block(in_features=int(32 * k),
                       out_features=int(32 * k),
                       norm_type=norm_type),
            SqueezeExciteBlock(in_features=int(32 * k), reduction=8))

        self.output1 = nn.Sequential(
            nn.Conv2d(int(32 * k), 3, kernel_size=(1, 1)), nn.Sigmoid())

        self.encode2_1 = nn.Sequential(
            conv_block(in_features=3,
                       out_features=int(32 * k),
                       norm_type=norm_type),
            conv_block(in_features=int(32 * k),
                       out_features=int(32 * k),
                       norm_type=norm_type),
            SqueezeExciteBlock(in_features=int(32 * k), reduction=8))
        self.encode2_2 = nn.Sequential(
            self.maxpool,
            conv_block(in_features=int(32 * k),
                       out_features=int(64 * k),
                       norm_type=norm_type),
            conv_block(in_features=int(64 * k),
                       out_features=int(64 * k),
                       norm_type=norm_type),
            SqueezeExciteBlock(in_features=int(64 * k), reduction=8))
        self.encode2_3 = nn.Sequential(
            self.maxpool,
            conv_block(in_features=int(64 * k),
                       out_features=int(128 * k),
                       norm_type=norm_type),
            conv_block(in_features=int(128 * k),
                       out_features=int(128 * k),
                       norm_type=norm_type),
            SqueezeExciteBlock(in_features=int(128 * k), reduction=8))
        self.encode2_4 = nn.Sequential(
            self.maxpool,
            conv_block(in_features=int(128 * k),
                       out_features=int(256 * k),
                       norm_type=norm_type),
            conv_block(in_features=int(256 * k),
                       out_features=int(256 * k),
                       norm_type=norm_type),
            SqueezeExciteBlock(in_features=int(256 * k), reduction=8))
        self.encode2_5 = self.maxpool

        self.ASPP_2 = DoubleASPP(in_features=int(256 * k),
                                 out_features=int(64 * k1),
                                 norm_type=norm_type)

        self.decode2_0 = self.upsample
        self.decode2_1 = nn.Sequential(
            conv_block(in_features=int(64 * k1) + 512 + int(256 * k),
                       out_features=int(256 * k),
                       norm_type=norm_type),
            conv_block(in_features=int(256 * k),
                       out_features=int(256 * k),
                       norm_type=norm_type),
            SqueezeExciteBlock(in_features=int(256 * k), reduction=8),
            self.upsample)
        self.decode2_2 = nn.Sequential(
            conv_block(in_features=int(256 * k) + 256 + int(128 * k),
                       out_features=int(128 * k),
                       norm_type=norm_type),
            conv_block(in_features=int(128 * k),
                       out_features=int(128 * k),
                       norm_type=norm_type),
            SqueezeExciteBlock(in_features=int(128 * k), reduction=8),
            self.upsample)
        self.decode2_3 = nn.Sequential(
            conv_block(in_features=int(128 * k) + 128 + int(64 * k),
                       out_features=int(64 * k),
                       norm_type=norm_type),
            conv_block(in_features=int(64 * k),
                       out_features=int(64 * k),
                       norm_type=norm_type),
            SqueezeExciteBlock(in_features=int(64 * k), reduction=8),
            self.upsample)
        self.decode2_4 = nn.Sequential(
            conv_block(in_features=int(64 * k) + 64 + int(32 * k),
                       out_features=int(32 * k),
                       norm_type=norm_type),
            conv_block(in_features=int(32 * k),
                       out_features=int(32 * k),
                       norm_type=norm_type),
            SqueezeExciteBlock(in_features=int(32 * k), reduction=8))

        self.output2 = nn.Conv2d(int(32 * k),
                                 out_features,
                                 kernel_size=(1, 1),
                                 padding=(0, 0))

        self.initialize_weights()

    def forward(self, x):
        x_in = self.normalize(x)
        # Encoder-Decoder-1
        x1_1 = self.vgg1(x_in)
        x1_2 = self.vgg2(x1_1)
        x1_3 = self.vgg3(x1_2)
        x1_4 = self.vgg4(x1_3)
        x = self.vgg5(x1_4)
        x = self.ASPP_1(x)
        x = self.decode1_0(x)
        x = self.decode1_1(torch.cat((x, x1_4), dim=1))
        x = self.decode1_2(torch.cat((x, x1_3), dim=1))
        x = self.decode1_3(torch.cat((x, x1_2), dim=1))
        x = self.decode1_4(torch.cat((x, x1_1), dim=1))
        x = self.output1(x)
        # Encoder-Decoder-2
        x *= x_in
        x2_1 = self.encode2_1(x)
        x2_2 = self.encode2_2(x2_1)
        x2_3 = self.encode2_3(x2_2)
        x2_4 = self.encode2_4(x2_3)
        x = self.encode2_5(x2_4)
        x = self.ASPP_2(x)
        x = self.decode2_0(x)
        x = self.decode2_1(torch.cat((x, x2_4, x1_4), dim=1))
        x = self.decode2_2(torch.cat((x, x2_3, x1_3), dim=1))
        x = self.decode2_3(torch.cat((x, x2_2, x1_2), dim=1))
        x = self.decode2_4(torch.cat((x, x2_1, x1_1), dim=1))
        x = self.output2(x)

        return x

    def normalize(self, x):
        return (x - self.mu) / self.sigma

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()


if __name__ == '__main__':

    def test(batchsize):
        in_channels = 3
        in1 = torch.rand(batchsize, in_channels, 512, 512).to('cuda')
        model = DoubleUnet(out_features=3, k=0.25, norm_type='bn').to('cuda')

        out1 = model(in1)
        total_params = sum(p.numel() for p in model.parameters())

        return out1.shape, total_params

    shape, total_params = test(batchsize=8)
    print('Shape : ', shape, '\nTotal params : ', total_params)
