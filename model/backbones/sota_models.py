__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import torch
from torch import nn
from torchvision import models
from model.main_blocks import Avgpool, conv_block
import functools
import operator

class VGG16(nn.Module):
    def __init__(self, pre_trained, req_grad, bn, out_channels=1, input_dim=(3, 512, 512)):
        nn.Module.__init__(self)
        self.adaptavgpool = nn.AdaptiveAvgPool2d(2)
        if bn:
            self.features = models.vgg16_bn(pretrained=pre_trained).features[:-1]

        else:
            self.features = models.vgg16(pretrained=pre_trained).features[:-1]

        for param in self.features.parameters():
            param.requires_grad = req_grad

        num_features_before_fcnn = functools.reduce(
            operator.mul,
            list(self.adaptavgpool(self.features(torch.rand(1, *input_dim))).shape))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features_before_fcnn, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=out_channels),
        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.features(x)
        out = self.adaptavgpool(out)
        out = out.view(batch_size, -1)  # flatten the vector
        out = self.classifier(out)
        return out


class VGG19(nn.Module):
    def __init__(self, pre_trained, req_grad, bn, out_channels=2, input_dim=(3, 512, 512)):
        nn.Module.__init__(self)
        self.adaptavgpool = nn.AdaptiveAvgPool2d(2)
        if bn:
            self.features = models.vgg19_bn(pretrained=pre_trained).features[:-1]

        else:
            self.features = models.vgg19(pretrained=pre_trained).features[:-1]

        for param in self.features.parameters():
            param.requires_grad = req_grad

        num_features_before_fcnn = functools.reduce(
            operator.mul,
            list(self.adaptavgpool(self.features(torch.rand(1, *input_dim))).shape))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features_before_fcnn, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=out_channels),
        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.features(x)
        out = self.adaptavgpool(out)
        out = out.view(batch_size, -1)  # flatten the vector
        out = self.classifier(out)
        return out


class DenseNet(nn.Module):
    def __init__(self, net_type, pre_trained, req_grad, out_channels=1, input_dim=(3, 512, 512)):
        nn.Module.__init__(self)
        if net_type == 'densenet-121':
            self.features = models.densenet121(pretrained=pre_trained).features

        if net_type == 'densenet-161':
            self.features = models.densenet161(pretrained=pre_trained).features

        if net_type == 'densenet-169':
            self.features = models.densenet169(pretrained=pre_trained).features

        if net_type == 'densenet-201':
            self.features = models.densenet201(pretrained=pre_trained).features

        for param in self.features.parameters():
            param.requires_grad = req_grad

        num_features_before_fcnn = functools.reduce(
            operator.mul,
            list(self.features(torch.rand(1, *input_dim)).shape))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features_before_fcnn, out_features=out_channels),

        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.features(x)
        out = out.view(batch_size, -1)  # flatten the vector
        out = self.classifier(out)
        return out


class ResNet(nn.Module):
    def __init__(self, net_type, pre_trained=False, req_grad=True, out_channels=1000):
        nn.Module.__init__(self)
        if net_type == 'ResNet-18':
            self.features = nn.Sequential(*(list(models.resnet18(pretrained=pre_trained).children())[:-2]))
            self.classifier = nn.Sequential(
                nn.Linear(in_features=512, out_features=out_channels),
            )

        if net_type == 'ResNet-34':
            self.features = nn.Sequential(*(list(models.resnet34(pretrained=pre_trained).children())[:-2]))
            self.classifier = nn.Sequential(
                nn.Linear(in_features=512, out_features=out_channels),
            )

        if net_type == 'ResNet-50':
            self.features = nn.Sequential(*(list(models.resnet50(pretrained=pre_trained).children())[:-2]))
            self.classifier = nn.Sequential(
                nn.Linear(in_features=2048, out_features=out_channels),
            )

        if net_type == 'ResNet-101':
            self.features = nn.Sequential(*(list(models.resnet101(pretrained=pre_trained).children())[:-2]))
            self.classifier = nn.Sequential(
                nn.Linear(in_features=2048, out_features=out_channels),
            )

        if net_type == 'ResNet-152':
            self.features = nn.Sequential(*(list(models.resnet152(pretrained=pre_trained).children())[:-2]))
            self.classifier = nn.Sequential(
                nn.Linear(in_features=2048, out_features=out_channels),
            )

        for param in self.features.parameters():
            param.requires_grad = req_grad

        if not pre_trained:
            self.initialize_weights()

        self.avgpool = Avgpool()


    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = self.classifier(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
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

if __name__ == '__main__':
    if __name__ == '__main__':
        def test(batchsize):
            in_channels = 3
            in1 = torch.rand(batchsize, in_channels, 224, 224)
            model = ResNet(net_type='ResNet-50')
            out1 = model(in1)
            print(model)
            total_params = sum(p.numel() for p in model.parameters())

            return out1.shape, total_params


        shape, total_params = test(batchsize=32)
        print('Shape : ', shape, '\nTotal params : ', total_params)
