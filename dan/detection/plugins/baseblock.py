import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from dan.classifier.baseblocks import *

from dan.design.builder import BACKBONES, PLUGINS


# # just combine classic basic operator to create functional module
# def conv_bn(inc, ouc, stride=1, leaky=0):
#     return nn.Sequential(nn.Conv2d(inc, ouc, 3, stride, 1, bias=False),
#                          nn.BatchNorm2d(ouc),
#                          nn.LeakyReLU(negative_slope=leaky, inplace=True))


# def conv_bn_no_relu(inc, ouc, stride):
#     return nn.Sequential(nn.Conv2d(inc, ouc, 3, stride, 1, bias=False),
#                          nn.BatchNorm2d(ouc))


# def conv_bn1x1(inc, ouc, stride, leaky=0):
#     return nn.Sequential(nn.Conv2d(inc, ouc, 1, stride, 0, bias=False),
#                          nn.BatchNorm2d(ouc),
#                          nn.LeakyReLU(negative_slope=leaky, inplace=True))


# def conv_dw(inc, ouc, stride, leaky=0.1):
#     return nn.Sequential(
#         nn.Conv2d(inc, inc, 3, stride, 1, groups=inc, bias=False),
#         nn.BatchNorm2d(inc), nn.LeakyReLU(negative_slope=leaky, inplace=True),
#         nn.Conv2d(inc, ouc, 1, 1, 0, bias=False), nn.BatchNorm2d(ouc),
#         nn.LeakyReLU(negative_slope=leaky, inplace=True))


# class BasicConv2d(nn.Module):
#     def __init__(self, inc, ouc, **kwargs):
#         super(BasicConv2d, self).__init__()
#         self.conv = nn.Conv2d(inc, ouc, bias=False, **kwargs)
#         self.bn = nn.BatchNorm2d(ouc, eps=1e-5)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return F.relu(x, inplace=True)


# # classic functional module, but just for some special network, not auto dynamic
# class Inception(nn.Module):
#     def __init__(self):
#         super(Inception, self).__init__()
#         self.branch1x1 = BasicConv2d(128, 32, kernel_size=1, padding=0)
#         self.branch1x1_2 = BasicConv2d(128, 32, kernel_size=1, padding=0)
#         self.branch3x3_reduce = BasicConv2d(128, 24, kernel_size=1, padding=0)
#         self.branch3x3 = BasicConv2d(24, 32, kernel_size=3, padding=1)
#         self.branch3x3_reduce_2 = BasicConv2d(128,
#                                               24,
#                                               kernel_size=1,
#                                               padding=0)
#         self.branch3x3_2 = BasicConv2d(24, 32, kernel_size=3, padding=1)
#         self.branch3x3_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)

#     def forward(self, x):
#         branch1x1 = self.branch1x1(x)

#         branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
#         branch1x1_2 = self.branch1x1_2(branch1x1_pool)

#         branch3x3_reduce = self.branch3x3_reduce(x)
#         branch3x3 = self.branch3x3(branch3x3_reduce)

#         branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
#         branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
#         branch3x3_3 = self.branch3x3_3(branch3x3_2)

#         outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_2]  # 32*4=128
#         return torch.cat(outputs, 1)


# class CRelu(nn.Module):
#     def __init__(self, inc, ouc, **kwargs):
#         super(CRelu, self).__init__()
#         self.conv = nn.Conv2d(inc, ouc, bias=False, **kwargs)
#         self.bn = nn.BatchNorm2d(ouc, eps=1e-5)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = torch.cat([x, -x], 1)
#         x = F.relu(x, inplace=True)
#         return x


@PLUGINS.register_module()
class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        # self.in_channel = in_channel
        # self.out_channel = out_channel
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel,
                                 out_channel // 4,
                                 stride=1,
                                 leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4,
                                         out_channel // 4,
                                         stride=1)

        self.conv7X7_2 = conv_bn(out_channel // 4,
                                 out_channel // 4,
                                 stride=1,
                                 leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4,
                                         out_channel // 4,
                                         stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


@BACKBONES.register_module()
class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky=0.1),  # 3
            conv_dw(8, 16, 1),  # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1),  # 59 + 32 = 91
            conv_dw(128, 128, 1),  # 91 + 32 = 123
            conv_dw(128, 128, 1),  # 123 + 32 = 155
            conv_dw(128, 128, 1),  # 155 + 32 = 187
            conv_dw(128, 128, 1),  # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),  # 219 +3 2 = 241
            conv_dw(256, 256, 1),  # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x