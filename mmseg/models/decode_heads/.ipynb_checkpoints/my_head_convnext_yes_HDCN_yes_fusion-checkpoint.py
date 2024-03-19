# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from ..builder import HEADS
from .decode_head import BaseDecodeHead
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn import constant_init, kaiming_init
from mmseg.ops import resize
from ..builder import NECKS
import math
from typing import Callable, Optional
from torch.nn.modules.utils import _pair
from torch.nn import init
from torch import Tensor
from mmcv.ops import CrissCrossAttention
@HEADS.register_module()
class MyHead(BaseDecodeHead):
    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(MyHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.getr = GETR()
        self.cls = nn.Conv2d(288, 2, kernel_size=1)

    def forward(self, inputs):
        output = self.getr(inputs)
        output = self.cls(output)
        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output
class AttentionModule(nn.Module):
    def __init__(self, nIn, d=1, KSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.bn_relu_2 = BNPReLU(nIn)
        self.conv1x1_1 = Conv(nIn, nIn // 4, KSize, 1, padding=1, bn_acti=True)

        self.dconv_4_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),
                              dilation=(d + 1, d + 1), groups=nIn // 16, bn_acti=True)

        self.dconv_4_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),
                              dilation=(d + 1, d + 1), groups=nIn // 16, bn_acti=True)

        self.dconv_4_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),
                              dilation=(d + 1, d + 1), groups=nIn // 16, bn_acti=True)

        self.dconv_1_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(1, 1),
                              dilation=(1, 1), groups=nIn // 16, bn_acti=True)

        self.dconv_1_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(1, 1),
                              dilation=(1, 1), groups=nIn // 16, bn_acti=True)

        self.dconv_1_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(1, 1),
                              dilation=(1, 1), groups=nIn // 16, bn_acti=True)

        self.dconv_2_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),
                              dilation=(int(d / 4 + 1), int(d / 4 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_2_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),
                              dilation=(int(d / 4 + 1), int(d / 4 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_2_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),
                              dilation=(int(d / 4 + 1), int(d / 4 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_3_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),
                              dilation=(int(d / 2 + 1), int(d / 2 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_3_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),
                              dilation=(int(d / 2 + 1), int(d / 2 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_3_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),
                              dilation=(int(d / 2 + 1), int(d / 2 + 1)), groups=nIn // 16, bn_acti=True)

        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        inp = self.bn_relu_1(input)
        inp = self.conv1x1_1(inp)

        o1_1 = self.dconv_1_1(inp)
        o1_2 = self.dconv_1_2(o1_1)
        o1_3 = self.dconv_1_3(o1_2)

        o2_1 = self.dconv_2_1(inp)
        o2_2 = self.dconv_2_2(o2_1)
        o2_3 = self.dconv_2_3(o2_2)

        o3_1 = self.dconv_3_1(inp)
        o3_2 = self.dconv_3_2(o3_1)
        o3_3 = self.dconv_3_3(o3_2)

        o4_1 = self.dconv_4_1(inp)
        o4_2 = self.dconv_4_2(o4_1)
        o4_3 = self.dconv_4_3(o4_2)

        output_1 = torch.cat([o1_1, o1_2, o1_3], 1)
        output_2 = torch.cat([o2_1, o2_2, o2_3], 1)
        output_3 = torch.cat([o3_1, o3_2, o3_3], 1)
        output_4 = torch.cat([o4_1, o4_2, o4_3], 1)

        ad1 = output_1
        ad2 = ad1 + output_2
        ad3 = ad2 + output_3
        ad4 = ad3 + output_4
        output = torch.cat([ad1, ad2, ad3, ad4], 1)
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        return output + input


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)
class GlobelBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_mul', )):
        super(GlobelBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))

            nn.init.kaiming_uniform_(self.channel_add_conv[0].weight, a=1)
            nn.init.kaiming_uniform_(self.channel_add_conv[3].weight, a=1)
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))

            nn.init.kaiming_uniform_(self.channel_mul_conv[0].weight, a=1)
            nn.init.kaiming_uniform_(self.channel_mul_conv[3].weight, a=1)
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out + out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class localBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.sigmoid_spatial = nn.Sigmoid()

        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv3[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv4[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv5[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv6[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv7[0].weight, a=1)

    def forward(self, x):
        res1 = x
        res2 = x
        x = self.conv1(x)
        x = x + self.conv2(x)
        x = self.conv3(x)
        x = x + self.conv4(x)
        x = self.conv5(x)
        x = x + self.conv6(x)
        x = self.conv7(x)
        x_mask = self.sigmoid_spatial(x)
        res1 = res1 * x_mask
        return res2 + res1


class ChannelMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels,h,w):
        super(ChannelMLP, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.fc1 = nn.Conv2d(in_channels,in_channels,1)
        self.fc2 = nn.Conv2d(in_channels,in_channels,1)
        self.fc3 = nn.Conv2d(in_channels,in_channels,1)
        nn.init.kaiming_uniform_(self.fc1.weight, a=1)
        nn.init.kaiming_uniform_(self.fc2.weight, a=1)
        nn.init.kaiming_uniform_(self.fc3.weight, a=1)

    def forward(self, x,h,w):
        # x = x.view(x.size(0), -1)  # 将每个通道的像素展平为向量形式
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        return x # 将输出张量重新调整为输入张量的形状
class DWConv2d(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, padding=1, bias=False):
        super(DWConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_chans, in_chans, kernel_size=kernel_size,
                                   padding=padding, groups=in_chans, bias=bias)
        self.pointwise = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=bias)

        nn.init.kaiming_uniform_(self.depthwise.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pointwise.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
class DeformableMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(DeformableMLP, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))  # kernel size == 1

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.offset_modulator_conv = DWConv2d(in_channels, 3 * in_channels)

        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        B, C, H, W = input.size()
        offset_modulator = self.offset_modulator_conv(input)
        offset_y, offset_x, modulator = torch.chunk(offset_modulator, 3, dim=1)
        modulator = 2. * torch.sigmoid(modulator)
        offset = torch.cat((offset_y, offset_x), dim=1)
        max_offset = max(H, W) // 4
        offset = offset.clamp(-max_offset, max_offset)
        x = torchvision.ops.deform_conv2d(input=input,
                                          offset=offset,
                                          weight=self.weight,
                                          bias=self.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation
                                          )

        x = self.act(self.norm(x))
        return x
class ConvFormer(nn.Module):
    def __init__(self, in_channels,out_channels,w,h):
        super(ConvFormer, self).__init__()
        self.layer_norm = nn.BatchNorm2d(in_channels)

        self.attention1 = AttentionModule(in_channels)

        self.localBlock = localBlock(in_features=in_channels,hidden_features=in_channels,out_features=in_channels)
        self.GlobelBlock = GlobelBlock(inplanes=in_channels,ratio=2)
        self.conv1 = nn.Conv2d(in_channels*2,in_channels,1)

        self.dense1 = ChannelMLP(in_channels,out_channels,w,h)
        self.dense2 = DeformableMLP(in_channels,in_channels)

        self.conv3_1 = nn.Conv2d(in_channels*2,in_channels,1)
        self.conv3_2 = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.conv4_1 = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.conv4_2 = nn.Conv2d(in_channels * 2, in_channels, 1)

        self.conv2 = nn.Conv2d(in_channels,out_channels,1)

    def forward(self, input_tensor,h,w):
        x = self.layer_norm(input_tensor)
        y2 = self.attention1(x)
        x1 = self.localBlock(x)
        y1 = self.GlobelBlock(y2)
        y1 = self.conv1(torch.cat([x1, y1], dim=1))

        x1 = self.conv3_1(torch.cat([input_tensor,y1],dim=1))
        x2 = self.conv3_2(torch.cat([x1-input_tensor,y2],dim=1))
        x = x1 + x2

        y3 = self.dense1(x,h,w)
        y4 = self.dense2(x)
        x3 = self.conv4_1(torch.cat([x, y3], dim=1))
        x4 = self.conv4_2(torch.cat([x3 - x, y4], dim=1))
        out_tensor = x3 + x4
        out_tensor = self.conv2(out_tensor)
        return out_tensor

class Attention2d(nn.Module):
    """ multi-head attention for 2D NCHW tensors"""
    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            dim_head: int = 32,
            bias: bool = True,
            expand_first: bool = True,
            rel_pos_cls: Callable = None,
            attn_drop: float = 0.,
            proj_drop: float = 0.
    ):
        super().__init__()
        dim_out = dim_out or dim
        dim_attn = dim_out if expand_first else dim
        self.num_heads = dim_attn // dim_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.qkv = nn.Conv2d(dim, dim_attn * 3, 1, bias=bias)
        self.rel_pos = rel_pos_cls(num_heads=self.num_heads) if rel_pos_cls else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim_attn, dim_out, 1, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor] = None):
        B, C, H, W = x.shape

        q, k, v = self.qkv(x).view(B, self.num_heads, self.dim_head * 3, -1).chunk(3, dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        if self.rel_pos is not None:
            attn = self.rel_pos(attn)
        elif shared_rel_pos is not None:
            attn = attn + shared_rel_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class GETR(nn.Module):

    def __init__(self):
        super().__init__()
        # self.rgb_input_proj = DCE(512, 256)
        self.encoder_tf_ss = Attention2d(dim=256,dim_out=256)

        self.encoder_shaper_7 = nn.Sequential(nn.Conv2d(512, 1024,1), nn.GELU())
        self.encoder_shaper_14 = nn.Sequential(nn.Conv2d(320, 256,1), nn.GELU())
        self.encoder_shaper_28 = nn.Sequential(nn.Conv2d(128, 64,1), nn.GELU())
        self.encoder_shaper_56 = nn.Sequential( nn.Conv2d(64, 16,1), nn.GELU())

        self.encoder_merge7_14 = nn.Sequential(nn.BatchNorm2d(512),
                                               nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=True),
                                               nn.LeakyReLU())
        self.encoder_merge28_14 = nn.Sequential(nn.BatchNorm2d(512),
                                                nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=True),
                                                nn.LeakyReLU())
        self.encoder_merge56_14 = nn.Sequential(nn.BatchNorm2d(512),
                                                nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=True),
                                                nn.LeakyReLU())
        self.fusion = nn.Conv2d(256, 256, 1)
        self.mask_head = MaskHeadSmallConv(512, [2048, 1024, 512, 256], 256)

        self.rgb_adapter3 = nn.Sequential(nn.Conv2d(512, 256, 1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(inplace=True))
        self.rgb_adapter2 = nn.Sequential(nn.Conv2d(320, 256, 1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(inplace=True))
        self.rgb_adapter1 = nn.Sequential(nn.Conv2d(128, 128, 1),
                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True))
        self.rgb_adapter0 = nn.Sequential(nn.Conv2d(64, 64, 1),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(inplace=True))
        self.lcfi_3 = ConvFormer(256, 256,16,16)
        self.lcfi_2 = ConvFormer(256, 256,32,32)
        self.lcfi_1 = ConvFormer(128, 128,64,64)
        self.lcfi_0 = ConvFormer(64, 64,128,128)
        self.fusion_module_3 = nn.Sequential(nn.Conv2d(448, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.fusion_module_2 = nn.Sequential(nn.Conv2d(448, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.fusion_module_1 = nn.Sequential(nn.Conv2d(576, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.fusion_module_0 = nn.Sequential(nn.Conv2d(640, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        nn.init.kaiming_uniform_(self.fusion_module_3[0].weight, a=1)
        nn.init.kaiming_uniform_(self.fusion_module_2[0].weight, a=1)
        nn.init.kaiming_uniform_(self.fusion_module_1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.fusion_module_0[0].weight, a=1)

    def forward(self, inputs):
        """Forward function."""
        # fusion_proj = self.rgb_input_proj(inputs[3])
        out_7s = self.encoder_shaper_7(inputs[3])
        out_7s = F.pixel_shuffle(out_7s, 2)

        out_14s = self.encoder_shaper_14(inputs[2])

        out_28s = self.encoder_shaper_28(inputs[1])
        out_28s = F.pixel_unshuffle(out_28s, 2)

        out_56s = self.encoder_shaper_56(inputs[0])
        out_56s = F.pixel_unshuffle(out_56s, 4)
        out = self.encoder_merge7_14(torch.cat([out_14s, out_7s], dim=1))
        out = self.encoder_merge28_14(torch.cat([out, out_28s], dim=1))
        out = self.encoder_merge56_14(torch.cat([out, out_56s], dim=1))
        out = self.encoder_tf_ss(out)
        fusion_memory = F.interpolate(out, size=inputs[3].shape[-2:], mode="bilinear")

        inputs[3] = self.rgb_adapter3(inputs[3])
        inputs[2] = self.rgb_adapter2(inputs[2])
        inputs[1] = self.rgb_adapter1(inputs[1])
        inputs[0] = self.rgb_adapter0(inputs[0])

        rgb_features_lcfi_3 = self.lcfi_3(inputs[3],16,16)
        rgb_features_lcfi_2 = self.lcfi_2(inputs[2],32,32)
        rgb_features_lcfi_1 = self.lcfi_1(inputs[1],64,64)
        rgb_features_lcfi_0 = self.lcfi_0(inputs[0],128,128)

        fusion_memory_3 = self.fusion_module_3(torch.cat(
            (F.interpolate(rgb_features_lcfi_2, size=inputs[3].shape[-2:], mode="bilinear"),
             F.interpolate(rgb_features_lcfi_1, size=inputs[3].shape[-2:], mode="bilinear"),
             F.interpolate(rgb_features_lcfi_0, size=inputs[3].shape[-2:], mode="bilinear")),1))
        fusion_memory_2 = self.fusion_module_2(torch.cat(
            (F.interpolate(rgb_features_lcfi_3, size=inputs[2].shape[-2:], mode="bilinear"),
             F.interpolate(rgb_features_lcfi_1, size=inputs[2].shape[-2:], mode="bilinear"),
             F.interpolate(rgb_features_lcfi_0, size=inputs[2].shape[-2:], mode="bilinear")),1))
        fusion_memory_1 = self.fusion_module_1(torch.cat(
            (F.interpolate(rgb_features_lcfi_3, size=inputs[1].shape[-2:], mode="bilinear"),
             F.interpolate(rgb_features_lcfi_2, size=inputs[1].shape[-2:], mode="bilinear"),
             F.interpolate(rgb_features_lcfi_0, size=inputs[1].shape[-2:], mode="bilinear")),1))
        fusion_memory_0 = self.fusion_module_0(torch.cat(
            (F.interpolate(rgb_features_lcfi_3, size=inputs[0].shape[-2:], mode="bilinear"),
             F.interpolate(rgb_features_lcfi_2, size=inputs[0].shape[-2:], mode="bilinear"),
             F.interpolate(rgb_features_lcfi_1, size=inputs[0].shape[-2:], mode="bilinear")),1))
        output = self.mask_head(fusion_memory,
                                [inputs[3], inputs[2],
                                 inputs[1], inputs[0]],
                                [fusion_memory_3, fusion_memory_2,
                                 fusion_memory_1, fusion_memory_0],
                                True)

        output = torch.cat((F.interpolate(out, size=output.shape[-2:], mode="bilinear"),output),1)
        return output

class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()
        # [512, 256, 128, 64, 32, 16]
        inter_dims = [dim, context_dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16]
        self.inference_module1 = InferenceModule(inter_dims[1], inter_dims[1])
        self.inference_module2 = InferenceModule(inter_dims[1], inter_dims[2])
        self.inference_module3 = InferenceModule(inter_dims[2], inter_dims[3])
        self.inference_module4 = InferenceModule(inter_dims[3], inter_dims[4])
        self.inference_module5 = InferenceModule(inter_dims[4], inter_dims[5])

        self.pa_module1 = PixelAttention(inter_dims[1], 3)
        self.pa_module2 = PixelAttention(inter_dims[1], 3)
        self.pa_module3 = PixelAttention(inter_dims[2], 3)
        self.pa_module4 = PixelAttention(inter_dims[3], 3)

        self.mask_out_conv = nn.Conv2d(5, 1, 1)
        self.edge_out_conv = nn.Conv2d(5, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fusion_memory, rgb_fpns, rgb_lcfi, gate_flag):

        x = fusion_memory

        x = self.inference_module1(self.pa_module1(x, rgb_fpns[0], rgb_lcfi[0], gate_flag))  # 1/32

        x = F.interpolate(x, size=rgb_fpns[1].shape[-2:], mode="bilinear")
        x = self.inference_module2(self.pa_module2(x, rgb_fpns[1], rgb_lcfi[1], gate_flag))  # 1/16

        x = F.interpolate(x, size=rgb_fpns[2].shape[-2:], mode="bilinear")
        x = self.inference_module3(self.pa_module3(x, rgb_fpns[2], rgb_lcfi[2], gate_flag))  # 1/8

        x = F.interpolate(x, size=rgb_fpns[3].shape[-2:], mode="bilinear")
        x = self.inference_module4(self.pa_module4(x, rgb_fpns[3], rgb_lcfi[3], gate_flag))  # 1/4

        return x


class PixelAttention(nn.Module):
    def __init__(self, inchannels, times):
        super(PixelAttention, self).__init__()
        self.mask_conv1 = nn.Sequential(nn.Conv2d(inchannels * 2, inchannels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.BatchNorm2d(inchannels),
                                        nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.BatchNorm2d(inchannels),
                                        nn.Conv2d(inchannels, inchannels, 1))
        self.mask_conv2 = nn.Sequential(nn.Conv2d(inchannels * 2, inchannels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.BatchNorm2d(inchannels),
                                        nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.BatchNorm2d(inchannels),
                                        nn.Conv2d(inchannels, inchannels, 1))

    def forward(self, x, rgb, lcfi, gate_flag):
        mask1 = self.mask_conv1(torch.cat([x, rgb], 1))
        mask2 = self.mask_conv2(torch.cat([mask1-x,lcfi],1))
        return mask1 + mask2

class InferenceModule(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, indim, outdim):
        super().__init__()

        self.conv_block = nn.Sequential(nn.Conv2d(indim, outdim, 3, padding=1),
                                        nn.BatchNorm2d(outdim),
                                        nn.ReLU(inplace=True))

        self.edge_conv = nn.Sequential(nn.Conv2d(outdim, outdim, 3, padding=1),
                                       nn.BatchNorm2d(outdim),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(outdim, outdim, 3, padding=1),
                                       nn.BatchNorm2d(outdim),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(outdim, outdim, 3, padding=1),
                                       nn.BatchNorm2d(outdim),
                                       nn.ReLU(inplace=True))
        self.mask_conv = nn.Sequential(nn.Conv2d(outdim * 2, outdim, 3, padding=1),
                                       nn.BatchNorm2d(outdim),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(outdim, outdim, 3, padding=1),
                                       nn.BatchNorm2d(outdim),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(outdim, outdim, 3, padding=1),
                                       nn.BatchNorm2d(outdim),
                                       nn.ReLU(inplace=True))
        self.out_mask_lay = torch.nn.Conv2d(outdim, 1, 3, padding=1)
        self.out_edge_lay = torch.nn.Conv2d(outdim, 1, 3, padding=1)

        nn.init.kaiming_uniform_(self.conv_block[0].weight, a=1)
        nn.init.kaiming_uniform_(self.edge_conv[0].weight, a=1)
        nn.init.kaiming_uniform_(self.edge_conv[3].weight, a=1)
        nn.init.kaiming_uniform_(self.edge_conv[6].weight, a=1)
        nn.init.kaiming_uniform_(self.mask_conv[0].weight, a=1)
        nn.init.kaiming_uniform_(self.mask_conv[3].weight, a=1)
        nn.init.kaiming_uniform_(self.mask_conv[6].weight, a=1)
        nn.init.kaiming_uniform_(self.out_mask_lay.weight, a=1)
        nn.init.kaiming_uniform_(self.out_edge_lay.weight, a=1)

    def forward(self, x):
        x = self.conv_block(x)
        edge_feature = self.edge_conv(x)
        x = self.mask_conv(torch.cat([edge_feature, x], 1))
        return x