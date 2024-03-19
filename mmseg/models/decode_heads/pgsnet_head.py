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
from torch.nn.modules.utils import _pair
from torch.nn import init
from torch import Tensor

@HEADS.register_module()
class MyHead(BaseDecodeHead):
    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(MyHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.getr = GETR()
        self.cls = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, inputs):
        output = self.getr(inputs)
        output = self.cls(output)
        return output
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

        nn.init.kaiming_uniform_(self.conv.weight, a=1)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class Fusion(nn.Module):
    def __init__(self, inchannels):
        super(Fusion, self).__init__()
        self.conv1 = BasicConv(inchannels*2,inchannels,1)
        self.conv2 = BasicConv(inchannels*2,inchannels,1)
        self.mask_conv1 = nn.Sequential(BasicConv(inchannels * 2, inchannels, kernel_size=3, stride=1, padding=1,relu=True,bn=True),
                                        BasicConv(inchannels, inchannels, kernel_size=3, stride=1, padding=1,relu=True,bn=True),
                                        nn.Conv2d(inchannels, inchannels, 1))
        self.mask_conv2 = nn.Sequential(BasicConv(inchannels * 2, inchannels, kernel_size=3, stride=1, padding=1,relu=True,bn=True),
                                        BasicConv(inchannels, inchannels, kernel_size=3, stride=1, padding=1,relu=True,bn=True),
                                        nn.Conv2d(inchannels, inchannels, 1))

    def forward(self, x, rgb, lcfi):
        mask1 = self.conv1(torch.cat([x, rgb], 1))
        mask1 = torch.sigmoid(mask1)
        high =rgb * mask1

        mask2 = self.conv2(torch.cat([x, lcfi], 1))
        mask2 = torch.sigmoid(mask2)
        low = lcfi * mask2
        mask1 = self.mask_conv1(torch.cat([x, high], 1))
        mask2 = self.mask_conv2(torch.cat([mask1-x,low],1))
        return mask1 + mask2
class multi_dilated_layer(nn.Module):
    def __init__(self, input_channels, dilation_rate=[6, 12, 18]):
        super(multi_dilated_layer, self).__init__()
        self.rates = dilation_rate
        self.layer1 = nn.Sequential(
            BasicConv(in_planes=input_channels,out_planes=input_channels//4,kernel_size=1,relu=True,bn=False),
            BasicConv(input_channels // 4, input_channels // 4, 1,relu=True,bn=False)
        )
        self.layer2 = nn.Sequential(
            BasicConv(input_channels, input_channels // 4, 3, padding=6, dilation=self.rates[0],relu=True,bn=False),
            BasicConv(input_channels // 4, input_channels // 4, 1,relu=True,bn=False),
        )
        self.layer3 = nn.Sequential(
            BasicConv(input_channels, input_channels // 4, 3, padding=12, dilation=self.rates[1],relu=True,bn=False),
            BasicConv(input_channels // 4, input_channels // 4, 1,relu=True,bn=False),
        )
        self.layer4 = nn.Sequential(
            BasicConv(input_channels, input_channels // 4, 3, padding=18, dilation=self.rates[2],relu=True,bn=False),
            BasicConv(input_channels // 4, input_channels // 4, 1,relu=True,bn=False),
        )
        self.concat_process = nn.Sequential(
            BasicConv(input_channels, 1024, 1,relu=True,bn=False),
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x)

        x4_cat = torch.cat((x1, x2, x3, x4), 1)
        return x4_cat

class _UpProjection(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(num_input_features, num_output_features,kernel_size=(5,1), stride=1, padding=(2,0), bias=False),
                                   nn.Conv2d(num_input_features, num_output_features,kernel_size=(1,5), stride=1, padding=(0,2), bias=False))
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = BasicConv(num_output_features, num_output_features,kernel_size=3, stride=1, padding=1, bias=False,relu=False,bn=True)

        self.conv2 = nn.Sequential(nn.Conv2d(num_input_features, num_output_features,kernel_size=(5,1), stride=1, padding=(2,0), bias=False),
                                   nn.Conv2d(num_input_features, num_output_features,kernel_size=(1,5), stride=1, padding=(0,2), bias=False))
        self.bn2 = nn.BatchNorm2d(num_output_features)

        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv1[1].weight, a=1)
        nn.init.kaiming_uniform_(self.conv2[1].weight, a=1)
    def forward(self, x, size):
        x = F.upsample(x, size=size, mode='bilinear')
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.conv1_2(x_conv1)
        bran2 = self.bn2(self.conv2(x))
        out = self.relu(bran1 + bran2)
        return out

class AttentionModule(nn.Module):
    def __init__(self, nIn, d=1, KSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = nn.Sequential(nn.BatchNorm2d(nIn,eps=1e-3),nn.PReLU(nIn))
        self.bn_relu_2 = nn.Sequential(nn.BatchNorm2d(nIn,eps=1e-3),nn.PReLU(nIn))
        self.conv1x1_1 = BasicConv(nIn, nIn // 4, KSize, 1, padding=1, relu=True,bn=True)

        self.dconv_4_1 = BasicConv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),dilation=(d + 1, d + 1), groups=nIn // 16, relu=True,bn=True)
        self.dconv_4_2 = BasicConv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),dilation=(d + 1, d + 1), groups=nIn // 16, relu=True,bn=True)
        self.dconv_4_3 = BasicConv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),dilation=(d + 1, d + 1), groups=nIn // 16, relu=True,bn=True)
        self.dconv_1_1 = BasicConv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(1, 1),dilation=(1, 1), groups=nIn // 16, relu=True,bn=True)
        self.dconv_1_2 = BasicConv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(1, 1),dilation=(1, 1), groups=nIn // 16, relu=True,bn=True)
        self.dconv_1_3 = BasicConv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(1, 1),dilation=(1, 1), groups=nIn // 16, relu=True,bn=True)
        self.dconv_2_1 = BasicConv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),dilation=(int(d / 4 + 1), int(d / 4 + 1)), groups=nIn // 16, relu=True,bn=True)
        self.dconv_2_2 = BasicConv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),dilation=(int(d / 4 + 1), int(d / 4 + 1)), groups=nIn // 16, relu=True,bn=True)
        self.dconv_2_3 = BasicConv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),dilation=(int(d / 4 + 1), int(d / 4 + 1)), groups=nIn // 16, relu=True,bn=True)
        self.dconv_3_1 = BasicConv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),dilation=(int(d / 2 + 1), int(d / 2 + 1)), groups=nIn // 16, relu=True,bn=True)
        self.dconv_3_2 = BasicConv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),dilation=(int(d / 2 + 1), int(d / 2 + 1)), groups=nIn // 16, relu=True,bn=True)
        self.dconv_3_3 = BasicConv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),dilation=(int(d / 2 + 1), int(d / 2 + 1)), groups=nIn // 16, relu=True,bn=True)
        self.conv1x1 = BasicConv(nIn, nIn, 1, 1, padding=0, relu=True,bn=False)

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


class DCE(nn.Module):  # DepthCorrelation Encoder
    def __init__(self, features, out_features, sizes=(1, 2, 3, 6)):
        super(DCE, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.ups = nn.ModuleList([_UpProjection(out_features // 2, out_features // 2) for i in range(4)])
        self.bottleneck = BasicConv(features // 4 * len(sizes), out_features // 2, kernel_size=3, padding=1, bias=False,relu=True,bn=False)
        self.multi_layers = multi_dilated_layer(features)
        self.fusion = BasicConv(in_planes=features // 4 * 5, out_planes=features, kernel_size=3, stride=1, padding=1,relu=True,bn=True)
    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features // 4, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        # pdb.set_trace()
        h, w = feats.size(2), feats.size(3)
        x4_cat = self.multi_layers(feats)  # 1024
        # pdb.set_trace()
        priors = [up(stage(feats), [h, w]) for (stage, up) in zip(self.stages, self.ups)]
        psp = self.bottleneck(torch.cat(priors, 1))
        fusion_feat = torch.cat((psp, x4_cat), 1)
        return self.fusion(fusion_feat)
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
        self.conv1 = BasicConv(in_features, hidden_features, 1, bias=False,relu=True,bn=True)
        self.conv2 = BasicConv(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False,bn=True,relu=True)
        self.conv3 = BasicConv(hidden_features, hidden_features, 1, bias=False,bn=True,relu=True)
        self.conv4 = BasicConv(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False,bn=True,relu=True)
        self.conv5 = BasicConv(hidden_features, hidden_features, 1, bias=False,bn=True,relu=True)
        self.conv6 = BasicConv(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False,bn=True,relu=True)
        self.conv7 = BasicConv(hidden_features, out_features, 1, bias=False,bn=True,relu=True)
        self.sigmoid_spatial = nn.Sigmoid()

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
    def __init__(self, in_channels, hidden_channels):
        super(ChannelMLP, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.fc1 = BasicConv(in_channels,in_channels,1,bn=False,relu=True)
        self.fc2 = BasicConv(in_channels,in_channels,1,bn=False,relu=True)
        self.fc3 = nn.Conv2d(in_channels,in_channels,1)
        nn.init.kaiming_uniform_(self.fc3.weight, a=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x # 将输出张量重新调整为输入张量的形状
class DWConv2d(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, padding=1, bias=False):
        super(DWConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_chans, in_chans, kernel_size=kernel_size,padding=padding, groups=in_chans, bias=bias)
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
    def __init__(self, in_channels,out_channels):
        super(ConvFormer, self).__init__()
        self.layer_norm = nn.BatchNorm2d(in_channels)

        self.attention1 = AttentionModule(in_channels)
        self.localBlock = localBlock(in_features=in_channels,hidden_features=in_channels,out_features=in_channels)
        self.GlobelBlock = GlobelBlock(inplanes=in_channels,ratio=2)
        self.conv1 = nn.Conv2d(in_channels*2,in_channels,1)


        self.dense1 = ChannelMLP(in_channels,out_channels)
        self.dense2 = DeformableMLP(in_channels,in_channels)

        self.Fusion1 = Fusion(in_channels)
        self.Fusion2 = Fusion(in_channels)

        self.conv2 = nn.Conv2d(in_channels,out_channels,1)

    def forward(self, input_tensor):
        x = self.layer_norm(input_tensor)
        x1 = self.localBlock(x)
        y1 = self.GlobelBlock(x1)
        y1 = torch.cat([x1, y1], dim=1)
        y1 = self.conv1(y1)
        y2 = self.attention1(x)
        x = self.Fusion1(input_tensor,y1,y2)

        y3 = self.dense1(x)
        y4 = self.dense2(x)
        out_tensor = self.Fusion2(x,y3,y4)
        out_tensor = self.conv2(out_tensor)
        return out_tensor

class GETR(nn.Module):

    def __init__(self):
        super().__init__()
        self.rgb_input_proj = DCE(512, 256)
        self.fusion = nn.Conv2d(512, 256, 1)
        self.mask_head = MaskHeadSmallConv(512, [2048, 1024, 512, 256], 256)

        self.lcfi_3 = ConvFormer(512, 256)
        self.lcfi_2 = ConvFormer(320, 256)
        self.lcfi_1 = ConvFormer(128, 128)
        self.lcfi_0 = ConvFormer(64, 64)

        self.fusion_module_3 = nn.Sequential(nn.Conv2d(448, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.fusion_module_2 = nn.Sequential(nn.Conv2d(448, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.fusion_module_1 = nn.Sequential(nn.Conv2d(576, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.fusion_module_0 = nn.Sequential(nn.Conv2d(640, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        nn.init.kaiming_uniform_(self.fusion.weight, a=1)
        nn.init.kaiming_uniform_(self.fusion_module_3[0].weight, a=1)
        nn.init.kaiming_uniform_(self.fusion_module_2[0].weight, a=1)
        nn.init.kaiming_uniform_(self.fusion_module_1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.fusion_module_0[0].weight, a=1)

    def forward(self, inputs):
        """Forward function."""
        fusion_proj = self.rgb_input_proj(inputs[3])
        fusion_memory = self.fusion(fusion_proj)
        rgb_features_lcfi_3 = self.lcfi_3(inputs[3])
        rgb_features_lcfi_2 = self.lcfi_2(inputs[2])
        rgb_features_lcfi_1 = self.lcfi_1(inputs[1])
        rgb_features_lcfi_0 = self.lcfi_0(inputs[0])


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
        return output

class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : channel attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out
class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : spatial attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out
class Positioning(nn.Module):
    def __init__(self, channel):
        super(Positioning, self).__init__()
        self.channel = channel
        self.cab = CA_Block(self.channel)
        self.sab = SA_Block(self.channel)
        self.map = nn.Conv2d(self.channel, 1, 7, 1, 3)

    def forward(self, x):
        cab = self.cab(x)
        sab = self.sab(cab)

        return sab
class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()
        # [512, 256, 128, 64, 32, 16]
        inter_dims = [dim, context_dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16]
        self.Positioning = Positioning(inter_dims[1])
        self.inference_module2 = InferenceModule(inter_dims[1], inter_dims[2])
        self.inference_module3 = InferenceModule(inter_dims[2], inter_dims[3])
        self.inference_module4 = InferenceModule(inter_dims[3], inter_dims[4])

        self.rgb_adapter1 = BasicConv(512, inter_dims[1], 1,bn=True,relu=True)
        self.rgb_adapter2 = BasicConv(320, inter_dims[1], 1,bn=True,relu=True)
        self.rgb_adapter3 = BasicConv(128, inter_dims[2], 1,bn=True,relu=True)
        self.rgb_adapter4 = BasicConv(64, inter_dims[3], 1,bn=True,relu=True)

        self.Fusion1 = Fusion(inter_dims[1])
        self.Fusion2 = Fusion(inter_dims[1])
        self.Fusion3 = Fusion(inter_dims[2])
        self.Fusion4 = Fusion(inter_dims[3])

        self.mask_out_conv = nn.Conv2d(5, 1, 1)
        self.edge_out_conv = nn.Conv2d(5, 1, 1)

    def forward(self, fusion_memory, rgb_fpns, rgb_lcfi, gate_flag):
        x = fusion_memory

        rgb_cur_fpn = self.rgb_adapter1(rgb_fpns[0])
        x = self.Positioning(self.Fusion1(x, rgb_cur_fpn, rgb_lcfi[0]))  # 1/32

        rgb_cur_fpn = self.rgb_adapter2(rgb_fpns[1])
        x = F.interpolate(x, size=rgb_cur_fpn.shape[-2:], mode="bilinear")
        x = self.inference_module2(self.Fusion2(x, rgb_cur_fpn, rgb_lcfi[1]))  # 1/16

        rgb_cur_fpn = self.rgb_adapter3(rgb_fpns[2])
        x = F.interpolate(x, size=rgb_cur_fpn.shape[-2:], mode="bilinear")
        x = self.inference_module3(self.Fusion3(x, rgb_cur_fpn, rgb_lcfi[2]))  # 1/8

        rgb_cur_fpn = self.rgb_adapter4(rgb_fpns[3])
        x = F.interpolate(x, size=rgb_cur_fpn.shape[-2:], mode="bilinear")
        x = self.inference_module4(self.Fusion4(x, rgb_cur_fpn, rgb_lcfi[3]))  # 1/4
        return x


class InferenceModule(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, indim, outdim):
        super().__init__()

        self.conv_block = BasicConv(indim, outdim, 3, padding=1,bn=True,relu=True)

        self.edge_conv = nn.Sequential(BasicConv(outdim, outdim, 3, padding=1,bn=True,relu=True),
                                       BasicConv(outdim, outdim, 3, padding=1,bn=True,relu=True),
                                       BasicConv(outdim, outdim, 3, padding=1,bn=True,relu=True))
        self.mask_conv = nn.Sequential(BasicConv(outdim * 2, outdim, 3, padding=1,bn=True,relu=True),
                                       BasicConv(outdim, outdim, 3, padding=1,bn=True,relu=True),
                                       BasicConv(outdim, outdim, 3, padding=1,bn=True,relu=True),)

    def forward(self, x):
        x = self.conv_block(x)
        edge_feature = self.edge_conv(x)
        x = self.mask_conv(torch.cat([edge_feature, x], 1))
        return x