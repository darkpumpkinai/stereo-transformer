#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

# Modified by Alex Showalter-Bucher(darkpumpkin.ai)
# -Added support for using instance normalization in lieu of batch normalization 05/03/2021

from torch import nn
from utilities.misc import NestedTensor


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    """Custom version of the pytorch resnet block to include instance norm"""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_instance_norm: bool = False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm2d if use_instance_norm else nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SppBackbone(nn.Module):
    """
    Contracting path of feature descriptor using Spatial Pyramid Pooling,
    SPP followed by PSMNet (https://github.com/JiaRenChang/PSMNet)
    """

    def __init__(self, use_instance_norm: bool = False):
        super(SppBackbone, self).__init__()
        self.use_instance_norm = use_instance_norm
        self.inplanes = 32
        self.in_conv = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2, bias=False),
                                     nn.InstanceNorm2d(16) if use_instance_norm else nn.BatchNorm2d(16),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
                                     nn.InstanceNorm2d(16) if use_instance_norm else nn.BatchNorm2d(16),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
                                     nn.InstanceNorm2d(32) if use_instance_norm else nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))  # 1/2

        self.resblock_1 = self._make_layer(BasicBlock, 64, 3, 2)  # 1/4
        self.resblock_2 = self._make_layer(BasicBlock, 128, 3, 2)  # 1/8

        self.branch1 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     nn.Conv2d(128, 32, kernel_size=1, bias=False),
                                     nn.InstanceNorm2d(32) if self.use_instance_norm else nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     nn.Conv2d(128, 32, kernel_size=1, bias=False),
                                     nn.InstanceNorm2d(32) if self.use_instance_norm else nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((4, 4), stride=(4, 4)),
                                     nn.Conv2d(128, 32, kernel_size=1, bias=False),
                                     nn.InstanceNorm2d(32) if self.use_instance_norm else nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((2, 2), stride=(2, 2)),
                                     nn.Conv2d(128, 32, kernel_size=1, bias=False),
                                     nn.InstanceNorm2d(32) if self.use_instance_norm else nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.InstanceNorm2d(planes * block.expansion) if self.use_instance_norm else nn.BatchNorm2d(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_instance_norm=self.use_instance_norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_instance_norm=self.use_instance_norm))
        return nn.Sequential(*layers)

    def forward(self, x: NestedTensor):
        """
        :param x: NestedTensor
        :return: list containing feature descriptors at different spatial resolution
                0: [2N, 3, H, W]
                1: [2N, C0, H//4, W//4]
                2: [2N, C1, H//8, W//8]
                3: [2N, C2, H//16, W//16]
        """
        _, _, h, w = x.left.shape

        src_stereo = torch.cat([x.left, x.right], dim=0)  # 2NxCxHxW

        # in conv
        output = self.in_conv(src_stereo)  # 1/2

        # res blocks
        output_1 = self.resblock_1(output)  # 1/4
        output_2 = self.resblock_2(output_1)  # 1/8

        # spp
        h_spp, w_spp = math.ceil(h / 16), math.ceil(w / 16)
        spp_1 = self.branch1(output_2)
        spp_1 = F.interpolate(spp_1, size=(h_spp, w_spp), mode='bilinear', align_corners=False)
        spp_2 = self.branch2(output_2)
        spp_2 = F.interpolate(spp_2, size=(h_spp, w_spp), mode='bilinear', align_corners=False)
        spp_3 = self.branch3(output_2)
        spp_3 = F.interpolate(spp_3, size=(h_spp, w_spp), mode='bilinear', align_corners=False)
        spp_4 = self.branch4(output_2)
        spp_4 = F.interpolate(spp_4, size=(h_spp, w_spp), mode='bilinear', align_corners=False)
        output_3 = torch.cat([spp_1, spp_2, spp_3, spp_4], dim=1)  # 1/16

        return [output_1, output_2, output_3]


def build_backbone(args):
    return SppBackbone(args.instance_norm)
