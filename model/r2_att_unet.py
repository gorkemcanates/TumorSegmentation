# -*- encoding: utf-8 -*-
# @File    :   XNET.py
# @Time    :   2021/05/30 22:48:27
# @Author  :   Gorkem Can Ates
# @Contact :   g.canates@gmail.com
# @Desc    :   None

from berries.model.base import BaseModel
import torch
import torch.nn as nn
from model.main_block import RRCNN_block, R2_Attention_block, up_conv


class R2AttU_Net(BaseModel):
    def __init__(self,
                 in_features=3,
                 out_features=3,
                 t=2,
                 k=1,
                 norm_type='bn'):
        nn.Module.__init__(self)

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=in_features,
                                  ch_out=int(64 * k),
                                  t=t,
                                  norm_type=norm_type)

        self.RRCNN2 = RRCNN_block(ch_in=int(64 * k),
                                  ch_out=int(128 * k),
                                  t=t,
                                  norm_type=norm_type)

        self.RRCNN3 = RRCNN_block(ch_in=int(128 * k),
                                  ch_out=int(256 * k),
                                  t=t,
                                  norm_type=norm_type)

        self.RRCNN4 = RRCNN_block(ch_in=int(256 * k),
                                  ch_out=int(512 * k),
                                  t=t,
                                  norm_type=norm_type)

        self.RRCNN5 = RRCNN_block(ch_in=int(512 * k),
                                  ch_out=int(1024 * k),
                                  t=t,
                                  norm_type=norm_type)

        self.Up5 = up_conv(ch_in=int(1024 * k),
                           ch_out=int(512 * k),
                           norm_type=norm_type)
        self.Att5 = R2_Attention_block(input_encoder=int(512 * k),
                                       input_decoder=int(512 * k),
                                       output_dim=int(256 * k),
                                       norm_type=norm_type)
        self.Up_RRCNN5 = RRCNN_block(ch_in=int(1024 * k),
                                     ch_out=int(512 * k),
                                     t=t,
                                     norm_type=norm_type)

        self.Up4 = up_conv(ch_in=int(512 * k),
                           ch_out=int(256 * k),
                           norm_type=norm_type)
        self.Att4 = R2_Attention_block(input_encoder=int(256 * k),
                                       input_decoder=int(256 * k),
                                       output_dim=int(128 * k),
                                       norm_type=norm_type)
        self.Up_RRCNN4 = RRCNN_block(ch_in=int(512 * k),
                                     ch_out=int(256 * k),
                                     t=t,
                                     norm_type=norm_type)

        self.Up3 = up_conv(ch_in=int(256 * k),
                           ch_out=int(128 * k),
                           norm_type=norm_type)
        self.Att3 = R2_Attention_block(input_encoder=int(128 * k),
                                       input_decoder=int(128 * k),
                                       output_dim=int(64 * k),
                                       norm_type=norm_type)
        self.Up_RRCNN3 = RRCNN_block(ch_in=int(256 * k),
                                     ch_out=int(128 * k),
                                     t=t,
                                     norm_type=norm_type)

        self.Up2 = up_conv(ch_in=int(128 * k),
                           ch_out=int(64 * k),
                           norm_type=norm_type)
        self.Att2 = R2_Attention_block(input_encoder=int(64 * k),
                                       input_decoder=int(64 * k),
                                       output_dim=int(32 * k),
                                       norm_type=norm_type)
        self.Up_RRCNN2 = RRCNN_block(ch_in=int(128 * k),
                                     ch_out=int(64 * k),
                                     t=t,
                                     norm_type=norm_type)

        self.Conv_1x1 = nn.Conv2d(int(64 * k),
                                  out_features,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

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
        model = R2AttU_Net(out_channels=3, k=0.25, norm_type='bn').to('cuda')

        out1 = model(in1)
        total_params = sum(p.numel() for p in model.parameters())

        return out1.shape, total_params

    shape, total_params = test(batchsize=6)
    print('Shape : ', shape, '\nTotal params : ', total_params)
