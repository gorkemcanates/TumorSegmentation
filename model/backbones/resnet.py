__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import torch
import torch.nn as nn
from model.backbones.main_blocks import conv_block, linear_block, Avgpool, Identity
from model.backbones.attention_blocks import SqueezeExciteBlock, CBAM, RecursiveSqueezeExciteBlock, SPSE, RGSE, A2Nework
import matplotlib.pyplot as plt
from model.backbones.main_blocks import visualize

class ResNet(nn.Module):
    def __init__(self,
                 backbone_model,
                 in_channels,
                 out_channels,
                 attention=None,
                 in_dim=None
                 ):
        super(ResNet, self).__init__()

        if in_dim is None:
            in_dim = [224, 224]
        if backbone_model == 'ResNet-18':
            self.model = ResidualNetwork(in_channels=in_channels,
                                         out_channels=out_channels,
                                         backbone=BottleNeck_short,
                                         layers=[2, 2, 2, 2],
                                         attention=attention,
                                         in_dim=in_dim
                                         )

        elif backbone_model == 'ResNet-34':
            self.model = ResidualNetwork(in_channels=in_channels,
                                         out_channels=out_channels,
                                         backbone=BottleNeck_short,
                                         layers=[3, 4, 6, 3],
                                         attention=attention,
                                         in_dim=in_dim
                                         )

        elif backbone_model == 'ResNet-50':
            self.model = ResidualNetwork(in_channels=in_channels,
                                         out_channels=out_channels,
                                         backbone=BottleNeck,
                                         layers=[3, 4, 6, 3],
                                         attention=attention,
                                         in_dim=in_dim
                                         )
        elif backbone_model == 'ResNet-101':
            self.model = ResidualNetwork(in_channels=in_channels,
                                         out_channels=out_channels,
                                         backbone=BottleNeck,
                                         layers=[3, 4, 23, 3],
                                         attention=attention,
                                         in_dim=in_dim
                                         )

        elif backbone_model == 'ResNet-152':
            self.model = ResidualNetwork(in_channels=in_channels,
                                         out_channels=out_channels,
                                         backbone=BottleNeck,
                                         layers=[3, 8, 36, 3],
                                         attention=attention,
                                         in_dim=in_dim
                                         )

        else:
            raise AttributeError



    def forward(self, x):
        return self.model(x)

class ResidualNetwork(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 backbone,
                 layers,
                 attention,
                 in_dim
                 ):
        super(ResidualNetwork, self).__init__()
        self.channel = 64

        self.entry_block = EntryBlock(in_channels=in_channels,
                                      out_channels=64)

        self.block1 = self._layer(block=backbone,
                                  channel=64,
                                  layer=layers[0],
                                  stride=(1, 1),
                                  attention=attention,
                                  in_dimx=in_dim[0]//4,
                                  in_dimy=in_dim[1]//4)
        self.block2 = self._layer(block=backbone,
                                  channel=128,
                                  layer=layers[1],
                                  stride=(2, 2),
                                  attention=attention,
                                  in_dimx=in_dim[0] // 8,
                                  in_dimy=in_dim[1] // 8)
        self.block3 = self._layer(block=backbone,
                                  channel=256,
                                  layer=layers[2],
                                  stride=(2, 2),
                                  attention=attention,
                                  in_dimx=in_dim[0] // 16,
                                  in_dimy=in_dim[1] // 16)

        self.block4 = self._layer(block=backbone,
                                  channel=512,
                                  layer=layers[3],
                                  stride=(2, 2),
                                  attention=attention,
                                  in_dimx=in_dim[0] // 32,
                                  in_dimy=in_dim[1] // 32)

        self.avgpool = Avgpool()
        if backbone == BottleNeck_short:
            self.fc = linear_block(in_features=512,
                                   out_features=out_channels)

        elif backbone == BottleNeck:
            self.fc = linear_block(in_features=2048,
                                   out_features=out_channels)

        else:
            raise AttributeError

        self.initialize_weights()


    def forward(self, x):
        x = self.entry_block(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x



    def _layer(self, block, channel, layer, stride, attention, in_dimx, in_dimy):

        if stride!= (1, 1) or channel * block.expansion != self.channel:

            self.downsample = conv_block(in_features=self.channel,
                                         out_features=int(channel * block.expansion),
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=stride,
                                         norm_type='bn',
                                         activation=False,
                                         use_bias=False)
        else:
            self.downsample = None

        layers = []
        layers.append(block(self.channel,
                            channel,
                            stride=stride,
                            downsample = self.downsample,
                            attention=attention,
                            in_dimx=in_dimx,
                            in_dimy=in_dimy))
        self.channel = int(channel * block.expansion)

        for _ in range(1, layer):
            layers.append(block(self.channel,
                                channel,
                                attention=attention,
                                in_dimx=in_dimx,
                                in_dimy=in_dimy))

        return nn.Sequential(*layers)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)




class EntryBlock(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=64
                 ):
        super(EntryBlock, self).__init__()

        self.block = conv_block(in_features=in_channels,
                                out_features=out_channels,
                                kernel_size=(7, 7),
                                stride=(2, 2),
                                padding=(3, 3),
                                norm_type='bn',
                                activation=True,
                                use_bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1),
                                    dilation=(1, 1),
                                    ceil_mode=False)

    def forward(self, x):
        x = self.block(x)
        x = self.maxpool(x)
        return x


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=(1, 1),
                 downsample=None,
                 attention=None,
                 in_dimx=None,
                 in_dimy=None):
        super(BottleNeck, self).__init__()

        self.block1 = conv_block(in_features=in_channels,
                                out_features=out_channels,
                                kernel_size=(1, 1),
                                stride=stride,
                                padding=(0, 0),
                                norm_type='bn',
                                activation=True,
                                use_bias=False)

        self.block2 = conv_block(in_features=out_channels,
                                out_features=out_channels,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=(1, 1),
                                norm_type='bn',
                                activation=True,
                                use_bias=False)

        self.block3 = conv_block(in_features=out_channels,
                                out_features=int(out_channels * self.expansion),
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                padding=(0, 0),
                                norm_type='bn',
                                activation=False,
                                use_bias=False)
        self.downsample = downsample

        if attention == 'SE':
            self.attention = SqueezeExciteBlock(in_features=int(out_channels * self.expansion))
        elif attention == 'CBAM':
            self.attention = CBAM(in_features=int(out_channels * self.expansion))
        elif attention == 'RecSE':
            self.attention = RecursiveSqueezeExciteBlock(in_features=int(out_channels * self.expansion),
                                                         rec=3)
        elif attention == 'SPSE':
            self.attention = SPSE(in_features=int(out_channels * self.expansion), H=in_dimx, W=in_dimy)

        elif attention == 'RGSE':
            self.attention = RGSE(in_features=int(out_channels * self.expansion), H=in_dimx, W=in_dimy)

        elif attention == 'A2':
            self.attention = A2Nework(in_features=int(out_channels * self.expansion),
                                      cm=int(out_channels * self.expansion) // 16,
                                      cn=int(out_channels * self.expansion) // 16)

        else:
            self.attention = Identity()

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        res = x
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.attention(out)
        if self.downsample is not None:
            res = self.downsample(x)
        out = out + res
        out = self.relu(out)
        return out

class BottleNeck_short(nn.Module):
    expansion = 1
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=(1, 1),
                 downsample=None,
                 attention=None,
                 in_dimx=None,
                 in_dimy=None):
        super(BottleNeck_short, self).__init__()

        self.block1 = conv_block(in_features=in_channels,
                                out_features=out_channels,
                                kernel_size=(3, 3),
                                stride=stride,
                                padding=(1, 1),
                                norm_type='bn',
                                activation=True,
                                use_bias=False)


        self.block2 = conv_block(in_features=out_channels,
                                out_features=int(out_channels * self.expansion),
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=(1, 1),
                                norm_type='bn',
                                activation=False,
                                use_bias=False)
        self.downsample = downsample

        if attention == 'SE':
            self.attention = SqueezeExciteBlock(in_features=int(out_channels * self.expansion))
        elif attention == 'CBAM':
            self.attention = CBAM(in_features=int(out_channels * self.expansion))
        elif attention == 'RecSE':
            self.attention = RecursiveSqueezeExciteBlock(in_features=int(out_channels * self.expansion),rec=3)
        elif attention == 'SPSE':
            self.attention = SPSE(in_features=int(out_channels * self.expansion), H=in_dimx, W=in_dimy)
        elif attention == 'RGSE':
            self.attention = RGSE(in_features=int(out_channels * self.expansion), H=in_dimx, W=in_dimy)
        elif attention == 'A2':
            self.attention = A2Nework(in_features=int(out_channels * self.expansion),
                                      cm=int(out_channels * self.expansion) // 16,
                                      cn=int(out_channels * self.expansion) // 16)

        else:
            self.attention = Identity()

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        res = x
        out = self.block1(x)
        out1 = self.block2(out)
        out = self.attention(out1)
        if self.downsample is not None:
            res = self.downsample(x)
        out = out + res
        out = self.relu(out)
        return out


if __name__ == '__main__':

    def test(batchsize):
        in_channels = 3
        in1 = torch.rand(batchsize, in_channels, 224, 224)
        model = ResNet(backbone_model='ResNet-50',
                       in_channels=in_channels,
                       out_channels=1000,
                       attention='RGSE',
                       in_dim=[224, 224]
                       )

        out1 = model(in1)
        total_params = sum(p.numel() for p in model.parameters())
        print(model)
        return out1.shape, total_params


    shape, total_params = test(batchsize=32)
    print('Shape : ', shape, '\nTotal params : ', total_params)
    






