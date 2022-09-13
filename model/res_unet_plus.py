__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import torch
import torch.nn as nn
from model.main_block_v2 import conv_block, ResConv, AttentionBlock, ResUASPP, SqueezeExciteBlock


class ResUnetPlus(nn.Module):
    def __init__(self,
                 in_features=3,
                 out_features=3,
                 k=1,
                 squeeze=False,
                 norm_type='bn',
                 upsample_type='bilinear'):
        nn.Module.__init__(self)
        self.input_layer = nn.Sequential(
            conv_block(in_features=in_features,
                       out_features=int(32 * k),
                       norm_type=norm_type),
            nn.Conv2d(int(32 * k), int(32 * k), kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_features, int(32 * k), kernel_size=3, padding=1))

        self.squeeze_excite1 = SqueezeExciteBlock(int(32 * k),
                                                  reduction=int(16 * k),
                                                  squeeze_flag=squeeze)

        self.residual_conv1 = ResConv(int(32 * k),
                                      int(64 * k),
                                      stride=2,
                                      padding=1,
                                      norm_type=norm_type)

        self.squeeze_excite2 = SqueezeExciteBlock(int(64 * k),
                                                  reduction=int(16 * k),
                                                  squeeze_flag=squeeze)

        self.residual_conv2 = ResConv(int(64 * k),
                                      int(128 * k),
                                      stride=2,
                                      padding=1,
                                      norm_type=norm_type)

        self.squeeze_excite3 = SqueezeExciteBlock(int(128 * k),
                                                  reduction=int(16 * k),
                                                  squeeze_flag=squeeze)

        self.residual_conv3 = ResConv(int(128 * k),
                                      int(256 * k),
                                      stride=2,
                                      padding=1,
                                      norm_type=norm_type)

        self.aspp_bridge = ResUASPP(int(256 * k),
                                    int(512 * k),
                                    norm_type=norm_type)

        self.attn1 = AttentionBlock(int(128 * k),
                                    int(512 * k),
                                    int(512 * k),
                                    norm_type=norm_type)
        self.upsample1 = nn.Upsample(scale_factor=2,
                                     mode=upsample_type,
                                     align_corners=False)
        self.up_residual_conv1 = ResConv(int(512 * k) + int(128 * k),
                                         int(256 * k),
                                         norm_type=norm_type)

        self.attn2 = AttentionBlock(int(64 * k),
                                    int(256 * k),
                                    int(256 * k),
                                    norm_type=norm_type)
        self.upsample2 = nn.Upsample(scale_factor=2,
                                     mode=upsample_type,
                                     align_corners=False)
        self.up_residual_conv2 = ResConv(int(256 * k) + int(64 * k),
                                         int(128 * k),
                                         norm_type=norm_type)

        self.attn3 = AttentionBlock(int(32 * k),
                                    int(128 * k),
                                    int(128 * k),
                                    norm_type=norm_type)
        self.upsample3 = nn.Upsample(scale_factor=2,
                                     mode=upsample_type,
                                     align_corners=False)
        self.up_residual_conv3 = ResConv(int(128 * k) + int(32 * k),
                                         int(64 * k),
                                         norm_type=norm_type)

        self.aspp_out = ResUASPP(int(64 * k), int(32 * k), norm_type=norm_type)

        self.output_layer = nn.Conv2d(int(32 * k),
                                      out_features,
                                      kernel_size=(1, 1))
        self.initialize_weights()

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)

        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':

    def test(batchsize):
        in_channels = 3
        in1 = torch.rand(batchsize, in_channels, 512, 512).to('cuda')
        model = ResUnetPlus(in_features=in_channels,
                            out_features=3,
                            k=1,
                            norm_type='gn').to('cuda')

        out1 = model(in1)
        total_params = sum(p.numel() for p in model.parameters())

        return out1.shape, total_params

    shape, total_params = test(batchsize=4)
    print('Shape : ', shape, '\nTotal params : ', total_params)
