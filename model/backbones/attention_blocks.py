__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model.backbones.main_blocks import visualize

class SqueezeExciteBlock(nn.Module):
    def __init__(self, in_features, reduction=16):
        nn.Module.__init__(self)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_features, int(in_features // reduction), bias=True),
                                nn.ReLU(),
                                nn.Linear(int(in_features // reduction), in_features, bias=True),
                                nn.Sigmoid())

    def forward(self, x):
        x_init = x
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x_init * y.expand_as(x_init)


class RecursiveSqueezeExciteBlock(nn.Module):
    def __init__(self, in_features, reduction=16, rec=3):
        nn.Module.__init__(self)
        self.rec = rec
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_features, int(in_features // reduction), bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(int(in_features // reduction), in_features, bias=True),
                                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        out = x
        for i in range(self.rec):
            x_init = out
            y = self.avgpool(out).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            out = x_init * y.expand_as(x_init)
        return out


class RecursiveSqueezeExciteBlock_V2(nn.Module):
    def __init__(self, in_features, reduction=16, rec=3):
        nn.Module.__init__(self)
        self.rec = rec
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_features, int(in_features // reduction), bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(int(in_features // reduction), in_features, bias=True),
                                )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        out = x
        y = self.avgpool(out).view(b, c)

        for i in range(self.rec):
            y = self.fc(y)

        y = y.view(b, c, 1, 1)
        out = x * y.expand_as(x)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_features, reduction=16, rec=1):
        super(ChannelAttention, self).__init__()
        self.rec = rec
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Linear(in_features, int(in_features // reduction), bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(int(in_features // reduction), in_features, bias=True),
                                )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        x1 = self.avgpool(x).view(b, c)
        x2 = self.maxpool(x).view(b, c)

        x1 = self.fc(x1)
        x2 = self.fc(x2)
        y = self.sigmoid(x1 + x2)
        y = y.view(b, c, 1, 1)
        out = x * y.expand_as(x)
        return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.cpool = ChannelPool()
        self.conv = nn.Conv2d(in_channels=2,
                              out_channels=1,
                              kernel_size=(7, 7),
                              padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.cpool(x)
        y = self.conv(y)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class CBAM(nn.Module):
    def __init__(self, in_features, reduction=16):
        super(CBAM, self).__init__()
        self.c_att = ChannelAttention(in_features=in_features,
                                      reduction=reduction)
        self.s_att = SpatialAttention()

    def forward(self, x):
        c_att = self.c_att(x)
        s_att = self.s_att(c_att)
        return s_att


class SPSE(nn.Module):
    def __init__(self, in_features, H, W, reduction=16, rate=None):
        super(SPSE, self).__init__()
        if rate is None:
            rate = [1, 2, 4]
        self.sp1 = nn.MaxPool2d(kernel_size=(H // rate[0], W // rate[0]),
                                stride=(H // rate[0], W // rate[0]))
        self.sp2 = nn.MaxPool2d(kernel_size=(H // rate[1], W // rate[1]),
                                stride=(H // rate[1], W // rate[1]))
        self.sp3 = nn.MaxPool2d(kernel_size=(H // rate[2], W // rate[2]),
                                stride=(H // rate[2], W // rate[2]))
        self.fusion = nn.ModuleList(nn.Linear(in_features=21, out_features=1, bias=True)
                                    for _ in range(in_features))
        self.fc = nn.Sequential(nn.Linear(in_features, int(in_features // reduction), bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(int(in_features // reduction), in_features, bias=True),
                                )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x_init = x
        b, c, _, _ = x.size()
        p1 = self.sp1(x)
        p2 = self.sp2(x)
        p3 = self.sp3(x)
        p = torch.cat((p1.view(p1.shape[0], p1.shape[1], p1.shape[2]*p1.shape[3], 1),
                       p2.view(p2.shape[0], p2.shape[1], p2.shape[2]*p2.shape[3], 1),
                       p3.view(p3.shape[0], p3.shape[1], p3.shape[2]*p3.shape[3], 1)), dim=2)
        f = []
        for i in range(self.fusion.__len__()):
            f.append(self.fusion[i](p[:, i, :, 0]))

        fused = torch.cat(f, dim=1)
        out = self.fc(fused).view(b, c, 1, 1)
        out = x_init * out.expand_as(x_init)
        return out

class RGSE(nn.Module):
    def __init__(self, in_features, H, W, K=7, reduction=16):
        super(RGSE, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=(K, K))

        self.fusion = nn.ModuleList(nn.Linear(in_features=(H // K) * (W // K), out_features=1, bias=True)
                                    for _ in range(in_features))
        self.fc = nn.Sequential(nn.Linear(in_features, int(in_features // reduction), bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(int(in_features // reduction), in_features, bias=True),
                                )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_init = x
        b, c, _, _ = x.size()
        p_init = self.avgpool(x)
        p = p_init.view(b, c, p_init.shape[2] * p_init.shape[3], 1)
        f = []
        for i in range(self.fusion.__len__()):
            f.append(self.fusion[i](p[:, i, :, 0]))

        fused = torch.cat(f, dim=1)
        out = self.fc(fused).view(b, c, 1, 1)
        out = x_init * out.expand_as(x_init)
        return out

class EcaNetwork(nn.Module):
    def __init__(self):
        super(EcaNetwork, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(in_channels=1,
                              out_channels=1,
                              kernel_size=(3, 3),
                              padding=(1, 1),
                              stride=(1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_init = x
        x = self.avgpool(x)
        x = self.conv(x.squeeze(3).permute(0, 2, 1))
        x = self.sigmoid(x).permute(0, 2, 1).unsqueeze(2)
        return x_init * x.expand_as(x_init)


class A2Nework(nn.Module):
    def __init__(self,
                 in_features,
                 cm,
                 cn):
        super(A2Nework, self).__init__()
        self.cm = cm
        self.cn = cn
        self.conv1 = nn.Conv2d(in_channels=in_features,
                               out_channels=cm,
                               kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=in_features,
                               out_channels=cn,
                               kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=in_features,
                               out_channels=cn,
                               kernel_size=(1, 1))

        self.conv_out = nn.Conv2d(in_channels=cm,
                                  out_channels=in_features,
                                  kernel_size=(1, 1))


    def forward(self, x):
        b, c, h, w = x.shape
        A = self.conv1(x)
        B = self.conv2(x)
        V = self.conv3(x)
        bpool_init = A.view(b, self.cm, -1)
        att_maps = F.softmax(B.view(b, self.cn, -1))
        att_vecs = F.softmax(V.view(b, self.cn, -1))
        gathered = torch.bmm(bpool_init, att_maps.permute(0, 2, 1))
        distributed = gathered.matmul(att_vecs).view(b, self.cm, h, w)
        out = self.conv_out(distributed)
        return x + out


if __name__ == '__main__':
    model = A2Nework(256, 128, 128)
    inp = torch.rand(32, 256, 7, 7)
    out = model(inp)
    print(out.shape)









