import torch
import torch.nn as nn
import math
import warnings
import torchvision
from torch.nn.modules.utils import _pair as to_2tuple
from mmseg.models.builder import BACKBONES
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)


class Mlp(BaseModule):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class StemConv(BaseModule):
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W

class GhostConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,padding, stride=1):
        super(GhostConv2d, self).__init__()
        # Split channels equally, keep shape for 1x1 Conv
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size, stride, padding=padding),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.cheap_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        a = self.primary_conv(x)
        b = self.cheap_conv(x)
        x = torch.cat((a, b), dim=1)
        return x
#可变性卷积
class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1):
        super(DeformConv2d, self).__init__()
        offset_channels = 2 * kernel_size * kernel_size
        self.conv_offset_1 = nn.Conv2d(in_channels, offset_channels, kernel_size=kernel_size, padding=padding)
        self.conv = torchvision.ops.DeformConv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=1)

    def forward(self, x):
        offset = self.conv_offset_1(x)
        return self.conv(x, offset)


def base():
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
#         self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
#
#         self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
#
#         self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
#         self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0(x)
#
#         attn_0 = self.conv0_1(attn)
#         attn_0 = self.conv0_2(attn_0)
#
#         attn_1 = self.conv1_1(attn)
#         attn_1 = self.conv1_2(attn_1)
#
#         attn_2 = self.conv2_1(attn)
#         attn_2 = self.conv2_2(attn_2)
#         attn = attn + attn_0 + attn_1 + attn_2
#
#         attn = self.conv3(attn)
#
#         return attn * u

def base5_15():
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0_1 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
#
#         self.conv0_2 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
#
#
#         self.conv1_1 = nn.Conv2d(dim, dim,(1, 3), padding=(0, 1),groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim,(3, 1), padding=(1, 0),groups=dim)
#         self.conv2_1 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
#         self.conv2_2 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
#
#         self.conv3_1 = nn.Conv2d(dim, dim, (1, 3), dilation=3, padding=(0, 3), groups=dim)
#         self.conv3_2 = nn.Conv2d(dim, dim, (3, 1), dilation=3, padding=(3, 0), groups=dim)
#         self.conv4_2 = nn.Conv2d(dim, dim, (1, 3), dilation=3, padding=(0, 3), groups=dim)
#         self.conv4_1 = nn.Conv2d(dim, dim, (3, 1), dilation=3, padding=(3, 0), groups=dim)
#
#         self.conv5_1 = nn.Conv2d(dim, dim, (1, 7), dilation=3, padding=(0, 9), groups=dim)
#         self.conv5_2 = nn.Conv2d(dim, dim, (7, 1), dilation=3, padding=(9, 0), groups=dim)
#         self.conv6_2 = nn.Conv2d(dim, dim, (1, 7), dilation=3, padding=(0, 9), groups=dim)
#         self.conv6_1 = nn.Conv2d(dim, dim, (7, 1), dilation=3, padding=(9, 0), groups=dim)
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0_1(x)
#         attn = self.conv0_2(attn)
#
#
#         attn_1 = self.conv1_1(attn)
#         attn_1 = self.conv1_2(attn_1)
#         attn_2 = self.conv2_1(attn + attn_1)
#         attn_2 = self.conv2_2(attn_2)
#
#         attn_3 = self.conv3_1(attn)
#         attn_3 = self.conv3_2(attn_3)
#         attn_4 = self.conv4_1(attn + attn_3)
#         attn_4 = self.conv4_2(attn_4)
#
#         attn_5 = self.conv5_1(attn)
#         attn_5 = self.conv5_2(attn_5)
#         attn_6 = self.conv6_1(attn + attn_5)
#         attn_6 = self.conv6_2(attn_6)
#
#
#         attn = attn  + attn_1 + attn_2 + attn_3 + attn_4 + attn_5 + attn_6
#
#         attn = self.conv3(attn)
#
#         return attn * u

def base5_16():
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0_1 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
#
#         self.conv0_2 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
#
#
#         self.conv1_1 = nn.Conv2d(dim, dim,(1, 3), padding=(0, 1),groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim,(3, 1), padding=(1, 0),groups=dim)
#
#         self.conv3_1 = nn.Conv2d(dim, dim, (1, 3), dilation=3, padding=(0, 3), groups=dim)
#         self.conv3_2 = nn.Conv2d(dim, dim, (3, 1), dilation=3, padding=(3, 0), groups=dim)
#         self.conv5_1 = nn.Conv2d(dim, dim, (1, 7), dilation=3, padding=(0, 9), groups=dim)
#         self.conv5_2 = nn.Conv2d(dim, dim, (7, 1), dilation=3, padding=(9, 0), groups=dim)
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0_1(x)
#         attn = self.conv0_2(attn)
#         attn_1 = self.conv1_1(attn)
#         attn_1 = self.conv1_2(attn_1)
#         attn_3 = self.conv3_1(attn)
#         attn_3 = self.conv3_2(attn_3)
#         attn_5 = self.conv5_1(attn)
#         attn_5 = self.conv5_2(attn_5)
#
#         attn = attn  + attn_1 + attn_3  + attn_5
#         attn = self.conv3(attn)
#
#         return attn * u

def base5_17():
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0_1 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
#
#         self.conv0_2 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
#
#
#         self.conv1_1 = nn.Conv2d(dim, dim,(1, 3), padding=(0, 1),groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim,(3, 1), padding=(1, 0),groups=dim)
#         self.conv2_1 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
#         self.conv2_2 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
#
#         self.conv3_1 = nn.Conv2d(dim, dim, (1, 3), dilation=3, padding=(0, 3), groups=dim)
#         self.conv3_2 = nn.Conv2d(dim, dim, (3, 1), dilation=3, padding=(3, 0), groups=dim)
#         self.conv4_2 = nn.Conv2d(dim, dim, (1, 3), dilation=3, padding=(0, 3), groups=dim)
#         self.conv4_1 = nn.Conv2d(dim, dim, (3, 1), dilation=3, padding=(3, 0), groups=dim)
#
#         self.conv5_1 = nn.Conv2d(dim, dim, (1, 7), dilation=3, padding=(0, 9), groups=dim)
#         self.conv5_2 = nn.Conv2d(dim, dim, (7, 1), dilation=3, padding=(9, 0), groups=dim)
#         self.conv6_2 = nn.Conv2d(dim, dim, (1, 7), dilation=3, padding=(0, 9), groups=dim)
#         self.conv6_1 = nn.Conv2d(dim, dim, (7, 1), dilation=3, padding=(9, 0), groups=dim)
#
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0_1(x)
#         attn = self.conv0_2(attn)
#         attn_1 = self.conv1_1(attn)
#         attn_1 = self.conv1_2(attn_1)
#         attn_2 = self.conv2_1(attn)
#         attn_2 = self.conv2_2(attn_2 )
#
#         attn_3 = self.conv3_1(attn)
#         attn_3 = self.conv3_2(attn_3)
#         attn_4 = self.conv4_1(attn)
#         attn_4 = self.conv4_2(attn_4)
#
#         attn_5 = self.conv5_1(attn)
#         attn_5 = self.conv5_2(attn_5)
#         attn_6 = self.conv6_1(attn)
#         attn_6 = self.conv6_2(attn_6)
#
#         attn = attn  + attn_1 + attn_2 + attn_3 + attn_4 + attn_5 + attn_6
#
#         attn = self.conv3(attn)
#
#         return attn * u

def base5_18():
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0_1 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
#
#         self.conv0_2 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
#
#
#         self.conv1_1 = nn.Conv2d(dim, dim,(1, 3), padding=(0, 1),groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim,(3, 1), padding=(1, 0),groups=dim)
#         self.conv2_1 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
#         self.conv2_2 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
#
#         self.conv3_1 = nn.Conv2d(dim, dim, (1, 3), dilation=3, padding=(0, 3), groups=dim)
#         self.conv3_2 = nn.Conv2d(dim, dim, (3, 1), dilation=3, padding=(3, 0), groups=dim)
#         self.conv4_2 = nn.Conv2d(dim, dim, (1, 3), dilation=3, padding=(0, 3), groups=dim)
#         self.conv4_1 = nn.Conv2d(dim, dim, (3, 1), dilation=3, padding=(3, 0), groups=dim)
#
#         self.conv5_1 = nn.Conv2d(dim, dim, (1, 7), dilation=3, padding=(0, 9), groups=dim)
#         self.conv5_2 = nn.Conv2d(dim, dim, (7, 1), dilation=3, padding=(9, 0), groups=dim)
#         self.conv6_2 = nn.Conv2d(dim, dim, (1, 7), dilation=3, padding=(0, 9), groups=dim)
#         self.conv6_1 = nn.Conv2d(dim, dim, (7, 1), dilation=3, padding=(9, 0), groups=dim)
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0_1(x)
#         attn = self.conv0_2(attn)
#
#
#         attn_1 = self.conv1_1(attn)
#         attn_1 = self.conv1_2(attn_1)
#         attn_2 = self.conv2_1(attn + attn_1)
#         attn_2 = self.conv2_2(attn_2)
#
#         attn_3 = self.conv3_1(attn + attn_2)
#         attn_3 = self.conv3_2(attn_3)
#         attn_4 = self.conv4_1(attn + attn_3)
#         attn_4 = self.conv4_2(attn_4)
#
#         attn_5 = self.conv5_1(attn + attn_4)
#         attn_5 = self.conv5_2(attn_5)
#         attn_6 = self.conv6_1(attn + attn_5)
#         attn_6 = self.conv6_2(attn_6)
#
#
#         attn = attn  + attn_1 + attn_2 + attn_3 + attn_4 + attn_5 + attn_6
#
#         attn = self.conv3(attn)
#
#         return attn * u

def base5_19():
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0_1 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
#
#         self.conv0_2 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
#
#
#         self.conv1_1 = nn.Conv2d(dim, dim,(1, 3), padding=(0, 1),groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim,(3, 1), padding=(1, 0),groups=dim)
#         self.conv2_1 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
#         self.conv2_2 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
#
#         self.conv3_1 = nn.Conv2d(dim, dim, (1, 3), dilation=3, padding=(0, 3), groups=dim)
#         self.conv3_2 = nn.Conv2d(dim, dim, (3, 1), dilation=3, padding=(3, 0), groups=dim)
#         self.conv4_2 = nn.Conv2d(dim, dim, (1, 3), dilation=3, padding=(0, 3), groups=dim)
#         self.conv4_1 = nn.Conv2d(dim, dim, (3, 1), dilation=3, padding=(3, 0), groups=dim)
#
#         self.conv5_1 = nn.Conv2d(dim, dim, (1, 7), dilation=3, padding=(0, 9), groups=dim)
#         self.conv5_2 = nn.Conv2d(dim, dim, (7, 1), dilation=3, padding=(9, 0), groups=dim)
#         self.conv6_2 = nn.Conv2d(dim, dim, (1, 7), dilation=3, padding=(0, 9), groups=dim)
#         self.conv6_1 = nn.Conv2d(dim, dim, (7, 1), dilation=3, padding=(9, 0), groups=dim)
#
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0_1(x)
#         attn = self.conv0_2(attn)
#
#         attn_6 = self.conv6_1(attn)
#         attn_6 = self.conv6_2(attn_6)
#         attn_5 = self.conv5_1(attn + attn_6)
#         attn_5 = self.conv5_2(attn_5)
#         attn_4 = self.conv4_1(attn + attn_5)
#         attn_4 = self.conv4_2(attn_4)
#         attn_3 = self.conv3_1(attn + attn_4)
#         attn_3 = self.conv3_2(attn_3)
#         attn_2 = self.conv2_1(attn + attn_3)
#         attn_2 = self.conv2_2(attn_2 )
#         attn_1 = self.conv1_1(attn + attn_2)
#         attn_1 = self.conv1_2(attn_1)
#
#         attn = attn  + attn_1 + attn_2 + attn_3 + attn_4 + attn_5 + attn_6
#
#         attn = self.conv3(attn)
#
#         return attn * u

def base5_20():
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0_1 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
#
#         self.conv0_2 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
#
#
#         self.conv1_1 = nn.Conv2d(dim, dim,(1, 3), padding=(0, 1),groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim,(3, 1), padding=(1, 0),groups=dim)
#         self.conv2_1 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
#         self.conv2_2 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
#
#         self.conv3_1 = nn.Conv2d(dim, dim, (1, 3), dilation=3, padding=(0, 3), groups=dim)
#         self.conv3_2 = nn.Conv2d(dim, dim, (3, 1), dilation=3, padding=(3, 0), groups=dim)
#         self.conv4_2 = nn.Conv2d(dim, dim, (1, 3), dilation=3, padding=(0, 3), groups=dim)
#         self.conv4_1 = nn.Conv2d(dim, dim, (3, 1), dilation=3, padding=(3, 0), groups=dim)
#
#         self.conv5_1 = nn.Conv2d(dim, dim, (1, 7), dilation=3, padding=(0, 9), groups=dim)
#         self.conv5_2 = nn.Conv2d(dim, dim, (7, 1), dilation=3, padding=(9, 0), groups=dim)
#         self.conv6_2 = nn.Conv2d(dim, dim, (1, 7), dilation=3, padding=(0, 9), groups=dim)
#         self.conv6_1 = nn.Conv2d(dim, dim, (7, 1), dilation=3, padding=(9, 0), groups=dim)
#
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0_1(x)
#         attn = self.conv0_2(attn)
#         attn_1 = self.conv1_1(attn)
#         attn_1 = self.conv1_2(attn_1)
#         attn_2 = self.conv2_1(attn_1)
#         attn_2 = self.conv2_2(attn_2 )
#
#         attn_3 = self.conv3_1(attn)
#         attn_3 = self.conv3_2(attn_3)
#         attn_4 = self.conv4_1(attn_3)
#         attn_4 = self.conv4_2(attn_4)
#
#         attn_5 = self.conv5_1(attn)
#         attn_5 = self.conv5_2(attn_5)
#         attn_6 = self.conv6_1(attn_5)
#         attn_6 = self.conv6_2(attn_6)
#
#         attn = attn + attn_2 + attn_4 + attn_6
#
#         attn = self.conv3(attn)
#
#         return attn * u

def base5_21():
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0_1 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
#
#         self.conv0_2 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
#
#
#         self.conv1_1 = nn.Conv2d(dim, dim,(1, 3), padding=(0, 1),groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim,(3, 1), padding=(1, 0),groups=dim)
#
#         self.conv3_1 = nn.Conv2d(dim, dim, (1, 3), dilation=2, padding=(0, 2), groups=dim)
#         self.conv3_2 = nn.Conv2d(dim, dim, (3, 1), dilation=2, padding=(2, 0), groups=dim)
#         self.conv4_1 = nn.Conv2d(dim, dim, (1, 3), dilation=3, padding=(0, 3), groups=dim)
#         self.conv4_2 = nn.Conv2d(dim, dim, (3, 1), dilation=3, padding=(3, 0), groups=dim)
#
#         self.conv5_1 = nn.Conv2d(dim, dim, (1, 7), dilation=2, padding=(0, 6), groups=dim)
#         self.conv5_2 = nn.Conv2d(dim, dim, (7, 1), dilation=2, padding=(6, 0), groups=dim)
#         self.conv6_2 = nn.Conv2d(dim, dim, (1, 7), dilation=3, padding=(0, 9), groups=dim)
#         self.conv6_1 = nn.Conv2d(dim, dim, (7, 1), dilation=3, padding=(9, 0), groups=dim)
#
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0_1(x)
#         attn = self.conv0_2(attn)
#         attn_1 = self.conv1_1(attn)
#         attn_1 = self.conv1_2(attn_1)
#
#         attn_3 = self.conv3_1(attn)
#         attn_3 = self.conv3_2(attn_3)
#         attn_4 = self.conv4_1(attn)
#         attn_4 = self.conv4_2(attn_4)
#
#         attn_5 = self.conv5_1(attn)
#         attn_5 = self.conv5_2(attn_5)
#         attn_6 = self.conv6_1(attn)
#         attn_6 = self.conv6_2(attn_6)
#
#         attn = attn  + attn_1 + attn_3 + attn_4 + attn_5 + attn_6
#
#         attn = self.conv3(attn)
#
#         return attn * u

def base5_22():#可变性卷积（未完成）
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0_1 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
#
#         self.conv0_2 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
#
#
#         self.conv1_1 = nn.Conv2d(dim, dim,(1, 3), padding=(0, 1),groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim,(3, 1), padding=(1, 0),groups=dim)
#         self.conv2_1 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
#         self.conv2_2 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
#
#         self.conv3_1 = nn.Conv2d(dim, dim, (1, 3), dilation=3, padding=(0, 3), groups=dim)
#         self.conv3_2 = nn.Conv2d(dim, dim, (3, 1), dilation=3, padding=(3, 0), groups=dim)
#         self.conv4_2 = nn.Conv2d(dim, dim, (1, 3), dilation=3, padding=(0, 3), groups=dim)
#         self.conv4_1 = nn.Conv2d(dim, dim, (3, 1), dilation=3, padding=(3, 0), groups=dim)
#
#         self.conv5_1 = nn.Conv2d(dim, dim, (1, 7), dilation=3, padding=(0, 9), groups=dim)
#         self.conv5_2 = nn.Conv2d(dim, dim, (7, 1), dilation=3, padding=(9, 0), groups=dim)
#         self.conv6_2 = nn.Conv2d(dim, dim, (1, 7), dilation=3, padding=(0, 9), groups=dim)
#         self.conv6_1 = nn.Conv2d(dim, dim, (7, 1), dilation=3, padding=(9, 0), groups=dim)
#         self.conv7 = dcnNet(dim,dim)
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0_1(x)
#         attn = self.conv0_2(attn)
#
#         attn_6 = self.conv6_1(attn)
#         attn_6 = self.conv6_2(attn_6)
#         attn_5 = self.conv5_1(attn)
#         attn_5 = self.conv5_2(attn_5)
#         attn_4 = self.conv4_1(attn)
#         attn_4 = self.conv4_2(attn_4)
#         attn_3 = self.conv3_1(attn)
#         attn_3 = self.conv3_2(attn_3)
#         attn_2 = self.conv2_1(attn)
#         attn_2 = self.conv2_2(attn_2 )
#         attn_1 = self.conv1_1(attn)
#         attn_1 = self.conv1_2(attn_1)
#         attn_7 = self.conv7(attn)
#         attn = attn  + attn_1 + attn_2 + attn_3 + attn_4 + attn_5 + attn_6 + attn_7
#
#         attn = self.conv3(attn)
#
#         return attn * u


def base5_23():
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
#         self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
#
#         self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
#
#         self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
#         self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
#
#         self.conv3_1 = nn.Conv2d(dim, dim, (1, 15), padding=(0, 7), groups=dim)
#         self.conv3_2 = nn.Conv2d(dim, dim, (15,1), padding=(7, 0), groups=dim)
#
#         self.conv4_1 = nn.Conv2d(dim, dim, (1, 25), padding=(0, 12), groups=dim)
#         self.conv4_2 = nn.Conv2d(dim, dim, (25, 1), padding=(12, 0), groups=dim)
#
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0(x)
#
#         attn_0 = self.conv0_1(attn)
#         attn_0 = self.conv0_2(attn_0)
#
#         attn_1 = self.conv1_1(attn)
#         attn_1 = self.conv1_2(attn_1)
#
#         attn_2 = self.conv2_1(attn)
#         attn_2 = self.conv2_2(attn_2)
#
#         attn_3 = self.conv3_1(attn)
#         attn_3 = self.conv3_2(attn_3)
#
#         attn_4 = self.conv4_1(attn)
#         attn_4 = self.conv4_2(attn_4)
#         attn = attn + attn_0 + attn_1 + attn_2 + attn_3 + attn_4
#
#         attn = self.conv3(attn)
#
#         return attn * u

def base5_24():
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0_1 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
#
#         self.conv0_2 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
#
#
#         self.conv1_1 = nn.Conv2d(dim, dim,(1, 3), padding=(0, 1),groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim,(3, 1), padding=(1, 0),groups=dim)
#         self.conv2_1 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
#         self.conv2_2 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
#
#         self.conv3_1 = nn.Conv2d(dim, dim, (1, 3), dilation=3, padding=(0, 3), groups=dim)
#         self.conv3_2 = nn.Conv2d(dim, dim, (3, 1), dilation=3, padding=(3, 0), groups=dim)
#         self.conv4_2 = nn.Conv2d(dim, dim, (1, 3), dilation=3, padding=(0, 3), groups=dim)
#         self.conv4_1 = nn.Conv2d(dim, dim, (3, 1), dilation=3, padding=(3, 0), groups=dim)
#
#         self.conv5_1 = nn.Conv2d(dim, dim, (1, 7), dilation=3, padding=(0, 9), groups=dim)
#         self.conv5_2 = nn.Conv2d(dim, dim, (7, 1), dilation=3, padding=(9, 0), groups=dim)
#         self.conv6_2 = nn.Conv2d(dim, dim, (1, 7), dilation=3, padding=(0, 9), groups=dim)
#         self.conv6_1 = nn.Conv2d(dim, dim, (7, 1), dilation=3, padding=(9, 0), groups=dim)
#
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0_1(x)
#         attn = self.conv0_2(attn)
#         attn_1 = self.conv1_1(attn)
#         attn_1 = self.conv1_2(attn_1)
#         attn_2 = self.conv2_1(attn)
#         attn_2 = self.conv2_2(attn_2 )
#
#         attn_3 = self.conv3_1(attn + attn_1 + attn_2)
#         attn_3 = self.conv3_2(attn_3)
#         attn_4 = self.conv4_1(attn + attn_1 + attn_2)
#         attn_4 = self.conv4_2(attn_4)
#
#         attn_5 = self.conv5_1(attn + attn_3 + attn_4)
#         attn_5 = self.conv5_2(attn_5)
#         attn_6 = self.conv6_1(attn + attn_3 + attn_4)
#         attn_6 = self.conv6_2(attn_6)
#
#         attn = attn  + attn_1 + attn_2 + attn_3 + attn_4 + attn_5 + attn_6
#
#         attn = self.conv3(attn)
#
#         return attn * u

def base5_25():
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0_1 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
#
#         self.conv0_2 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
#
#
#         self.conv1_1 = nn.Conv2d(dim, dim,(1, 3), padding=(0, 1),groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim,(3, 1), padding=(1, 0),groups=dim)
#         self.conv2_1 = nn.Conv2d(dim, dim, (1, 3), dilation=(0, 3), padding=(0, 3), groups=dim)
#         self.conv2_2 = nn.Conv2d(dim, dim, (3, 1), dilation=(3, 0), padding=(3, 0), groups=dim)
#
#         self.conv3_1 = nn.Conv2d(dim, dim, (1, 3), dilation=(0, 5), padding=(0, 5), groups=dim)
#         self.conv3_2 = nn.Conv2d(dim, dim, (3, 1), dilation=(5, 0), padding=(5, 0), groups=dim)
#         self.conv4_1 = nn.Conv2d(dim, dim, (1, 3), dilation=(0, 7), padding=(0, 7), groups=dim)
#         self.conv4_2 = nn.Conv2d(dim, dim, (3, 1), dilation=(7, 0), padding=(7, 0), groups=dim)
#
#         self.conv5_1 = nn.Conv2d(dim, dim, (1, 3), dilation=(0, 9), padding=(0, 9), groups=dim)
#         self.conv5_2 = nn.Conv2d(dim, dim, (3, 1), dilation=(9, 0), padding=(9, 0), groups=dim)
#         self.conv6_1 = nn.Conv2d(dim, dim, (1, 3), dilation=(0, 11), padding=(0, 11), groups=dim)
#         self.conv6_2 = nn.Conv2d(dim, dim, (3, 1), dilation=(11, 0), padding=(11, 0), groups=dim)
#
#         self.conv7_1 = nn.Conv2d(dim, dim, (1, 3), dilation=(0, 13), padding=(0, 13), groups=dim)
#         self.conv7_2 = nn.Conv2d(dim, dim, (3, 1), dilation=(13, 0), padding=(13, 0), groups=dim)
#         self.conv8_1 = nn.Conv2d(dim, dim, (1, 3), dilation=(0, 15), padding=(0, 15), groups=dim)
#         self.conv8_2 = nn.Conv2d(dim, dim, (3, 1), dilation=(15, 0), padding=(15, 0), groups=dim)
#
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0_1(x)
#         attn = self.conv0_2(attn)
#         attn_1 = self.conv1_1(attn)
#         attn_1 = self.conv1_2(attn_1)
#         attn_2 = self.conv2_1(attn)
#         attn_2 = self.conv2_2(attn_2)
#
#         attn_3 = self.conv3_1(attn)
#         attn_3 = self.conv3_2(attn_3)
#         attn_4 = self.conv4_1(attn)
#         attn_4 = self.conv4_2(attn_4)
#
#         attn_5 = self.conv5_1(attn)
#         attn_5 = self.conv5_2(attn_5)
#         attn_6 = self.conv6_1(attn)
#         attn_6 = self.conv6_2(attn_6)
#
#         attn_7 = self.conv7_1(attn)
#         attn_7 = self.conv7_2(attn_7)
#         attn_8 = self.conv8_1(attn)
#         attn_8 = self.conv8_2(attn_8)
#
#         attn = attn  + attn_1 + attn_2 + attn_3 + attn_4 + attn_5 + attn_6 + attn_7 + attn_8
#
#         attn = self.conv3(attn)
#
#         return attn * u


def base5_26():
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0_1 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
#
#         self.conv0_2 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
#
#
#         self.conv1_1 = nn.Conv2d(dim, dim,(1, 3), padding=(0, 1),groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim,(3, 1), padding=(1, 0),groups=dim)
#         self.conv2_1 = nn.Conv2d(dim, dim, (1, 3), dilation=(0, 3), padding=(0, 3), groups=dim)
#         self.conv2_2 = nn.Conv2d(dim, dim, (3, 1), dilation=(3, 0), padding=(3, 0), groups=dim)
#
#         self.conv3_1 = nn.Conv2d(dim, dim, (1, 5), dilation=(0, 3), padding=(0, 6), groups=dim)
#         self.conv3_2 = nn.Conv2d(dim, dim, (5, 1), dilation=(3, 0), padding=(6, 0), groups=dim)
#         self.conv4_1 = nn.Conv2d(dim, dim, (1, 5), dilation=(0, 5), padding=(0, 10), groups=dim)
#         self.conv4_2 = nn.Conv2d(dim, dim, (5, 1), dilation=(5, 0), padding=(10, 0), groups=dim)
#
#         self.conv5_1 = nn.Conv2d(dim, dim, (1, 7), dilation=(0, 3), padding=(0, 9), groups=dim)
#         self.conv5_2 = nn.Conv2d(dim, dim, (7, 1), dilation=(3, 0), padding=(9, 0), groups=dim)
#         self.conv6_1 = nn.Conv2d(dim, dim, (1, 7), dilation=(0, 5), padding=(0, 15), groups=dim)
#         self.conv6_2 = nn.Conv2d(dim, dim, (7, 1), dilation=(5, 0), padding=(15, 0), groups=dim)
#
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0_1(x)
#         attn = self.conv0_2(attn)
#         attn_1 = self.conv1_1(attn)
#         attn_1 = self.conv1_2(attn_1)
#         attn_2 = self.conv2_1(attn)
#         attn_2 = self.conv2_2(attn_2)
#
#         attn_3 = self.conv3_1(attn)
#         attn_3 = self.conv3_2(attn_3)
#         attn_4 = self.conv4_1(attn)
#         attn_4 = self.conv4_2(attn_4)
#
#         attn_5 = self.conv5_1(attn)
#         attn_5 = self.conv5_2(attn_5)
#         attn_6 = self.conv6_1(attn)
#         attn_6 = self.conv6_2(attn_6)
#
#         attn = attn  + attn_1 + attn_2 + attn_3 + attn_4 + attn_5 + attn_6
#
#         attn = self.conv3(attn)
#
#         return attn * u

def base5_27():
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0_1 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
#
#         self.conv0_2 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
#
#
#         self.conv1_1 = nn.Conv2d(dim, dim,(1, 3), padding=(0, 1),groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim,(3, 1), padding=(1, 0),groups=dim)
#         self.conv2_1 = nn.Conv2d(dim, dim, (1, 3), dilation=(0, 3), padding=(0, 3), groups=dim)
#         self.conv2_2 = nn.Conv2d(dim, dim, (3, 1), dilation=(3, 0), padding=(3, 0), groups=dim)
#
#         self.conv3_1 = nn.Conv2d(dim, dim, (1, 5), dilation=(0, 3), padding=(0, 6), groups=dim)
#         self.conv3_2 = nn.Conv2d(dim, dim, (5, 1), dilation=(3, 0), padding=(6, 0), groups=dim)
#         self.conv4_1 = nn.Conv2d(dim, dim, (1, 5), dilation=(0, 5), padding=(0, 10), groups=dim)
#         self.conv4_2 = nn.Conv2d(dim, dim, (5, 1), dilation=(5, 0), padding=(10, 0), groups=dim)
#
#         self.conv5_1 = nn.Conv2d(dim, dim, (1, 7), dilation=(0, 3), padding=(0, 9), groups=dim)
#         self.conv5_2 = nn.Conv2d(dim, dim, (7, 1), dilation=(3, 0), padding=(9, 0), groups=dim)
#         self.conv6_1 = nn.Conv2d(dim, dim, (1, 7), dilation=(0, 5), padding=(0, 15), groups=dim)
#         self.conv6_2 = nn.Conv2d(dim, dim, (7, 1), dilation=(5, 0), padding=(15, 0), groups=dim)
#
#         self.conv3 = nn.Conv2d(dim*7, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0_1(x)
#         attn = self.conv0_2(attn)
#         attn_1 = self.conv1_1(attn)
#         attn_1 = self.conv1_2(attn_1)
#         attn_2 = self.conv2_1(attn)
#         attn_2 = self.conv2_2(attn_2)
#
#         attn_3 = self.conv3_1(attn)
#         attn_3 = self.conv3_2(attn_3)
#         attn_4 = self.conv4_1(attn)
#         attn_4 = self.conv4_2(attn_4)
#
#         attn_5 = self.conv5_1(attn)
#         attn_5 = self.conv5_2(attn_5)
#         attn_6 = self.conv6_1(attn)
#         attn_6 = self.conv6_2(attn_6)
#
#         attn = torch.cat([attn,attn_1,attn_2,attn_3,attn_4,attn_5,attn_6],dim=1)
#
#         attn = self.conv3(attn)
#
#         return attn * u

def base5_28():
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0_1 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
#
#         self.conv0_2 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
#
#
#         self.conv1_1 = nn.Conv2d(dim, dim,(1, 3), padding=(0, 1),groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim,(3, 1), padding=(1, 0),groups=dim)
#         self.conv2_1 = nn.Conv2d(dim, dim, (1, 3), dilation=(0, 3), padding=(0, 3), groups=dim)
#         self.conv2_2 = nn.Conv2d(dim, dim, (3, 1), dilation=(3, 0), padding=(3, 0), groups=dim)
#
#         self.conv3_1 = nn.Conv2d(dim, dim, (1, 5), dilation=(0, 3), padding=(0, 6), groups=dim)
#         self.conv3_2 = nn.Conv2d(dim, dim, (5, 1), dilation=(3, 0), padding=(6, 0), groups=dim)
#         self.conv4_1 = nn.Conv2d(dim, dim, (1, 7), dilation=(0, 3), padding=(0, 9), groups=dim)
#         self.conv4_2 = nn.Conv2d(dim, dim, (7, 1), dilation=(3, 0), padding=(9, 0), groups=dim)
#
#         self.conv5_1 = nn.Conv2d(dim, dim, (1, 9), dilation=(0, 3), padding=(0, 12), groups=dim)
#         self.conv5_2 = nn.Conv2d(dim, dim, (9, 1), dilation=(3, 0), padding=(12, 0), groups=dim)
#         self.conv6_1 = nn.Conv2d(dim, dim, (1, 11), dilation=(0, 3), padding=(0, 15), groups=dim)
#         self.conv6_2 = nn.Conv2d(dim, dim, (11, 1), dilation=(3, 0), padding=(15, 0), groups=dim)
#
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0_1(x)
#         attn = self.conv0_2(attn)
#
#
#         attn_1 = self.conv1_1(attn)
#         attn_1 = self.conv1_2(attn_1)
#         attn_2 = self.conv2_1(attn + attn_1)
#         attn_2 = self.conv2_2(attn_2)
#
#         attn_3 = self.conv3_1(attn + attn_2)
#         attn_3 = self.conv3_2(attn_3)
#         attn_4 = self.conv4_1(attn + attn_3)
#         attn_4 = self.conv4_2(attn_4)
#
#         attn_5 = self.conv5_1(attn + attn_4)
#         attn_5 = self.conv5_2(attn_5)
#         attn_6 = self.conv6_1(attn + attn_5)
#         attn_6 = self.conv6_2(attn_6)
#
#         attn = attn  + attn_1 + attn_2 + attn_3 + attn_4 + attn_5 + attn_6
#
#         attn = self.conv3(attn)
#
#         return attn * u



def base5_29():
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0_1 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
#
#         self.conv0_2 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
#
#
#         self.conv1_1 = nn.Conv2d(dim, dim,(1, 3), padding=(0, 1),groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim,(3, 1), padding=(1, 0),groups=dim)
#         self.conv2_1 = nn.Conv2d(dim, dim, (1, 3), dilation=(0, 3), padding=(0, 3), groups=dim)
#         self.conv2_2 = nn.Conv2d(dim, dim, (3, 1), dilation=(3, 0), padding=(3, 0), groups=dim)
#
#         self.conv3_1 = nn.Conv2d(dim, dim, (1, 5), dilation=(0, 3), padding=(0, 6), groups=dim)
#         self.conv3_2 = nn.Conv2d(dim, dim, (5, 1), dilation=(3, 0), padding=(6, 0), groups=dim)
#         self.conv4_1 = nn.Conv2d(dim, dim, (1, 7), dilation=(0, 3), padding=(0, 9), groups=dim)
#         self.conv4_2 = nn.Conv2d(dim, dim, (7, 1), dilation=(3, 0), padding=(9, 0), groups=dim)
#
#         self.conv5_1 = nn.Conv2d(dim, dim, (1, 9), dilation=(0, 3), padding=(0, 12), groups=dim)
#         self.conv5_2 = nn.Conv2d(dim, dim, (9, 1), dilation=(3, 0), padding=(12, 0), groups=dim)
#         self.conv6_1 = nn.Conv2d(dim, dim, (1, 11), dilation=(0, 3), padding=(0, 15), groups=dim)
#         self.conv6_2 = nn.Conv2d(dim, dim, (11, 1), dilation=(3, 0), padding=(15, 0), groups=dim)
#
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#         self.conv2 = nn.Conv2d(dim * 2, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0_1(x)
#         attn = self.conv0_2(attn)
#
#         attn_1 = self.conv1_1(attn)
#         attn_1 = self.conv1_2(attn_1)
#         attn_1_1 = self.conv2(torch.cat([attn,attn_1],dim=1))
#         attn_2 = self.conv2_1(attn_1_1)
#         attn_2 = self.conv2_2(attn_2)
#
#         attn_2_2 = self.conv2(torch.cat([attn,attn_2],dim=1))
#         attn_3 = self.conv3_1(attn_2_2)
#         attn_3 = self.conv3_2(attn_3)
#         attn_3_3 = self.conv2(torch.cat([attn,attn_3],dim=1))
#         attn_4 = self.conv4_1(attn_3_3)
#         attn_4 = self.conv4_2(attn_4)
#
#
#         attn_4_4 = self.conv2(torch.cat([attn,attn_4],dim=1))
#         attn_5 = self.conv5_1(attn_4_4)
#         attn_5 = self.conv5_2(attn_5)
#         attn_5_5 = self.conv2(torch.cat([attn,attn_5],dim=1))
#         attn_6 = self.conv6_1(attn_5_5)
#         attn_6 = self.conv6_2(attn_6)
#
#         attn = attn  + attn_1 + attn_2 + attn_3 + attn_4 + attn_5 + attn_6
#
#         attn = self.conv3(attn)
#
#         return attn * u

def base6_1():
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0_1 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
#
#         self.conv0_2 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
#
#
#         self.conv1_1 = DeformConv2d(dim,dim,kernel_size=3)
#         self.conv2_1 = DeformConv2d(dim,dim,kernel_size=3)
#
#         self.conv4_1 = nn.Conv2d(dim, dim, (1, 7) ,padding=(0, 3), groups=dim)
#         self.conv4_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
#
#         self.conv5_1 = nn.Conv2d(dim, dim, (1, 9), dilation=(0, 3), padding=(0, 12), groups=dim)
#         self.conv5_2 = nn.Conv2d(dim, dim, (9, 1), dilation=(3, 0), padding=(12, 0), groups=dim)
#         self.conv6_1 = nn.Conv2d(dim, dim, (1, 11), dilation=(0, 3), padding=(0, 15), groups=dim)
#         self.conv6_2 = nn.Conv2d(dim, dim, (11, 1), dilation=(3, 0), padding=(15, 0), groups=dim)
#
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0_1(x)
#         attn = self.conv0_2(attn)
#
#
#         attn_1 = self.conv1_1(attn)
#         attn_2 = self.conv2_1(attn)
#         attn_4 = self.conv4_1(attn)
#         attn_4 = self.conv4_2(attn_4)
#
#         attn_5 = self.conv5_1(attn)
#         attn_5 = self.conv5_2(attn_5)
#         attn_6 = self.conv6_1(attn)
#         attn_6 = self.conv6_2(attn_6)
#
#         attn = attn + attn_1 + attn_2  + attn_4 + attn_5 + attn_6
#
#         attn = self.conv3(attn)
#
#         return attn * u

def base6_2():
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0_1 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
#
#         self.conv0_2 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
#
#
#         self.conv1_1 = GhostConv2d(dim,dim,kernel_size=3)
#         self.conv2_1 = GhostConv2d(dim,dim,kernel_size=3)
#
#         self.conv4_1 = nn.Conv2d(dim, dim, (1, 7) ,padding=(0, 3), groups=dim)
#         self.conv4_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
#
#         self.conv5_1 = nn.Conv2d(dim, dim, (1, 9), dilation=(0, 3), padding=(0, 12), groups=dim)
#         self.conv5_2 = nn.Conv2d(dim, dim, (9, 1), dilation=(3, 0), padding=(12, 0), groups=dim)
#         self.conv6_1 = nn.Conv2d(dim, dim, (1, 11), dilation=(0, 3), padding=(0, 15), groups=dim)
#         self.conv6_2 = nn.Conv2d(dim, dim, (11, 1), dilation=(3, 0), padding=(15, 0), groups=dim)
#
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0_1(x)
#         attn = self.conv0_2(attn)
#
#
#         attn_1 = self.conv1_1(attn)
#         attn_2 = self.conv2_1(attn)
#         attn_4 = self.conv4_1(attn)
#         attn_4 = self.conv4_2(attn_4)
#
#         attn_5 = self.conv5_1(attn)
#         attn_5 = self.conv5_2(attn_5)
#         attn_6 = self.conv6_1(attn)
#         attn_6 = self.conv6_2(attn_6)
#
#         attn = attn + attn_1 + attn_2  + attn_4 + attn_5 + attn_6
#
#         attn = self.conv3(attn)
#
#         return attn * u

def base6_3():
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0_1 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
#
#         self.conv0_2 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
#
#
#         self.conv1_1 = DeformConv2d(dim,dim,kernel_size=3,padding=1)
#         self.conv2_1 = DeformConv2d(dim,dim,kernel_size=5,padding=2)
#
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0_1(x)
#         attn = self.conv0_2(attn)
#
#
#         attn_1 = self.conv1_1(attn)
#         attn_2 = self.conv2_1(attn)
#
#
#         attn = attn + attn_1 + attn_2
#
#         attn = self.conv3(attn)
#
#         return attn * u

def base6_4():
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0_1 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
#
#         self.conv0_2 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
#
#         self.conv1_1 = DeformConv2d(dim, dim, kernel_size=3, padding=1)
#         self.conv2_1 = nn.Conv2d(dim, dim,(1, 3), padding=(0, 1),groups=dim)
#         self.conv2_2 = nn.Conv2d(dim, dim,(3, 1), padding=(1, 0),groups=dim)
#         self.conv3_1 = nn.Conv2d(dim, dim, (1, 3), dilation=(0, 3), padding=(0, 3), groups=dim)
#         self.conv3_2 = nn.Conv2d(dim, dim, (3, 1), dilation=(3, 0), padding=(3, 0), groups=dim)
#
#         self.conv4_1 = nn.Conv2d(dim, dim, (1, 7), dilation=(0, 3), padding=(0, 9), groups=dim)
#         self.conv4_2 = nn.Conv2d(dim, dim, (7, 1), dilation=(3, 0), padding=(9, 0), groups=dim)
#
#         self.conv6_1 = nn.Conv2d(dim, dim, (1, 11), dilation=(0, 3), padding=(0, 15), groups=dim)
#         self.conv6_2 = nn.Conv2d(dim, dim, (11, 1), dilation=(3, 0), padding=(15, 0), groups=dim)
#
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0_1(x)
#         attn = self.conv0_2(attn)
#
#
#         attn_1 = self.conv1_1(attn)
#         attn_2 = self.conv2_1(attn + attn_1)
#         attn_2 = self.conv2_2(attn_2)
#
#         attn_3 = self.conv3_1(attn + attn_2)
#         attn_3 = self.conv3_2(attn_3)
#         attn_4 = self.conv4_1(attn + attn_3)
#         attn_4 = self.conv4_2(attn_4)
#
#         attn_6 = self.conv6_1(attn + attn_4)
#         attn_6 = self.conv6_2(attn_6)
#
#         attn = attn  + attn_1 + attn_2 + attn_3 + attn_4 + attn_6
#
#         attn = self.conv3(attn)
#
#         return attn * u

def base6_5():
    return 1
# class AttentionModule(BaseModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.activation = nn.GELU()
#         self.conv = nn.Conv2d(dim*2, dim, 3, padding=1, groups=dim)
#
#         self.conv0_1 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
#
#         self.conv0_2 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
#
#
#         self.conv1_1 = nn.Conv2d(dim, dim,(1, 3), padding=(0, 1),groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim,(3, 1), padding=(1, 0),groups=dim)
#         self.conv2_1 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
#         self.conv2_2 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
#
#         self.conv3_1 = nn.Conv2d(dim, dim, (1, 3), dilation=3, padding=(0, 3), groups=dim)
#         self.conv3_2 = nn.Conv2d(dim, dim, (3, 1), dilation=3, padding=(3, 0), groups=dim)
#         self.conv4_2 = nn.Conv2d(dim, dim, (1, 3), dilation=3, padding=(0, 3), groups=dim)
#         self.conv4_1 = nn.Conv2d(dim, dim, (3, 1), dilation=3, padding=(3, 0), groups=dim)
#
#         self.conv5_1 = nn.Conv2d(dim, dim, (1, 7), dilation=3, padding=(0, 9), groups=dim)
#         self.conv5_2 = nn.Conv2d(dim, dim, (7, 1), dilation=3, padding=(9, 0), groups=dim)
#         self.conv6_2 = nn.Conv2d(dim, dim, (1, 7), dilation=3, padding=(0, 9), groups=dim)
#         self.conv6_1 = nn.Conv2d(dim, dim, (7, 1), dilation=3, padding=(9, 0), groups=dim)
#
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0_1(x)
#         attn = self.conv0_2(attn)
#
#         attn_1 = self.conv1_1(attn)
#         attn_1 = self.conv1_2(attn_1)
#         attn_1 = self.activation(attn_1)
#         attn_2 = self.conv2_1(attn)
#         attn_2 = self.conv2_2(attn_2 )
#         attn_2 = self.activation(attn_2)
#
#         attn_12 = torch.cat([attn_1,attn_2],dim=1)
#         attn_12 = self.conv(attn_12)
#         attn_12 = self.activation(attn_12)
#
#         attn_3 = self.conv3_1(attn + attn_12)
#         attn_3 = self.conv3_2(attn_3)
#         attn_3 = self.activation(attn_3)
#         attn_4 = self.conv4_1(attn + attn_12)
#         attn_4 = self.conv4_2(attn_4)
#         attn_4 = self.activation(attn_4)
#
#         attn_34 = torch.cat([attn_3,attn_4],dim=1)
#         attn_34 = self.conv(attn_34)
#         attn_34 = self.activation(attn_34)
#
#         attn_5 = self.conv5_1(attn + attn_34)
#         attn_5 = self.conv5_2(attn_5)
#         attn_5 = self.activation(attn_5)
#         attn_6 = self.conv6_1(attn + attn_34)
#         attn_6 = self.conv6_2(attn_6)
#         attn_6 = self.activation(attn_6)
#         attn_56 = torch.cat([attn_5,attn_6],dim=1)
#         attn_56 = self.conv(attn_56)
#         attn_56 = self.activation(attn_56)
#
#         attn = attn  + attn_12 + attn_34 + attn_56
#
#         attn = self.conv3(attn)
#
#         return attn * u

def base6_6():
    return 1
class AttentionModule(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.activation = nn.GELU()
        self.conv = nn.Conv2d(dim*2, dim, 3, padding=1, groups=dim)

        self.conv0_1 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)

        self.conv0_2 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)


        self.conv1_1 = nn.Conv2d(dim, dim,(1, 3), padding=(0, 1),groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim,(3, 1), padding=(1, 0),groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)

        self.conv3_1 = nn.Conv2d(dim, dim, (1, 3), dilation=3, padding=(0, 3), groups=dim)
        self.conv3_2 = nn.Conv2d(dim, dim, (3, 1), dilation=3, padding=(3, 0), groups=dim)
        self.conv4_2 = nn.Conv2d(dim, dim, (1, 3), dilation=3, padding=(0, 3), groups=dim)
        self.conv4_1 = nn.Conv2d(dim, dim, (3, 1), dilation=3, padding=(3, 0), groups=dim)

        self.conv5_1 = nn.Conv2d(dim, dim, (1, 7), dilation=3, padding=(0, 9), groups=dim)
        self.conv5_2 = nn.Conv2d(dim, dim, (7, 1), dilation=3, padding=(9, 0), groups=dim)
        self.conv6_2 = nn.Conv2d(dim, dim, (1, 7), dilation=3, padding=(0, 9), groups=dim)
        self.conv6_1 = nn.Conv2d(dim, dim, (7, 1), dilation=3, padding=(9, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0_1(x)
        attn = self.conv0_2(attn)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2 )

        attn_12 = torch.cat([attn_1,attn_2],dim=1)
        attn_12 = self.conv(attn_12)
        attn_12 = self.activation(attn_12)

        attn_3 = self.conv3_1(attn + attn_12)
        attn_3 = self.conv3_2(attn_3)
        attn_4 = self.conv4_1(attn + attn_12)
        attn_4 = self.conv4_2(attn_4)

        attn_34 = torch.cat([attn_3,attn_4],dim=1)
        attn_34 = self.conv(attn_34)
        attn_34 = self.activation(attn_34)

        attn_5 = self.conv5_1(attn + attn_34)
        attn_5 = self.conv5_2(attn_5)
        attn_6 = self.conv6_1(attn + attn_34)
        attn_6 = self.conv6_2(attn_6)
        attn_56 = torch.cat([attn_5,attn_6],dim=1)
        attn_56 = self.conv(attn_56)
        attn_56 = self.activation(attn_56)

        attn = attn  + attn_12 + attn_34 + attn_56

        attn = self.conv3(attn)

        return attn * u



class SpatialAttention(BaseModule):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(BaseModule):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(BaseModule):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W


@ BACKBONES.register_module()
class MSCAN(BaseModule):
    def __init__(self,
                 in_chans=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 pretrained=None,
                 init_cfg=None):
        super(MSCAN, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0], norm_cfg=norm_cfg)
            else:
                patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                                stride=4 if i == 0 else 2,
                                                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                                embed_dim=embed_dims[i],
                                                norm_cfg=norm_cfg)

            block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i],
                                         drop=drop_rate, drop_path=dpr[cur + j],
                                         norm_cfg=norm_cfg)
                                   for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self):
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:

            super(MSCAN, self).init_weights()

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x
