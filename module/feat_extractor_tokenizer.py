#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

# Modified by Alex Showalter-Bucher(darkpumpkin.ai)
# -Added support for using instance normalization in lieu of batch normalization 05/03/2021

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List
from utilities.misc import center_crop
import torch.utils.checkpoint as cp


class DenseLayer(nn.Module):
    """Custom version of the torchvision DenseLayer which utilizes instance norm or batch norm"""

    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        memory_efficient: bool = False,
        use_instance_norm: bool = False
    ) -> None:
        super(DenseLayer, self).__init__()

        norm_layer = nn.InstanceNorm2d if use_instance_norm else nn.BatchNorm2d

        self.norm1: norm_layer
        self.add_module('norm1', norm_layer(num_input_features))
        self.relu1: nn.ReLU
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.conv1: nn.Conv2d
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False))

        self.norm2: norm_layer
        self.add_module('norm2', norm_layer(bn_size * growth_rate))
        self.relu2: nn.ReLU
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.conv2: nn.Conv2d
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class DenseBlock(nn.ModuleDict):
    """Custom version of the torchvision DenseBlock which utilizes use either instance norm
    or batch norm"""

    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
        use_instance_norm: bool = False
    ) -> None:
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                use_instance_norm=use_instance_norm
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class TransitionUp(nn.Module):
    """
    Scale the resolution up by transposed convolution
    """

    def __init__(self, in_channels: int, out_channels: int, scale: int = 2):
        super().__init__()

        self.convTrans = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=0, bias=True)

    def forward(self, x: Tensor, skip: Tensor):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class DoubleConv(nn.Module):
    """
    Two conv2d-bn-relu modules
    """

    def __init__(self, in_channels: int, out_channels: int, use_instance_norm: bool = False):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels) if use_instance_norm else nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels) if use_instance_norm else nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Tokenizer(nn.Module):
    """
    Expanding path of feature descriptor using DenseBlocks
    """

    def __init__(self, block_config: int, hidden_dim: int, backbone_feat_channel: list, growth_rate: int,
                 use_instance_norm: bool = False):
        super(Tokenizer, self).__init__()

        backbone_feat_channel.reverse()  # reverse so we have high-level first (lowest-spatial res)
        self.block_config = block_config
        self.growth_rate = growth_rate
        self.hidden_dim = hidden_dim

        # 1/16
        self.bottle_neck = DenseBlock(block_config, backbone_feat_channel[0], 4, drop_rate=0.0,
                                       growth_rate=growth_rate, use_instance_norm=use_instance_norm)
        up = []
        dense_block = []
        prev_block_channels = growth_rate * block_config

        # 1/8
        up.append(TransitionUp(prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + backbone_feat_channel[1]
        dense_block.append(DenseBlock(block_config, cur_channels_count, 4, drop_rate=0.0, growth_rate=growth_rate,
                                      use_instance_norm=use_instance_norm))
        prev_block_channels = growth_rate * block_config

        # 1/4
        up.append(TransitionUp(prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + backbone_feat_channel[2]
        dense_block.append(DenseBlock(block_config, cur_channels_count, 4, drop_rate=0.0, growth_rate=growth_rate,
                                      use_instance_norm=use_instance_norm))

        self.up = nn.ModuleList(up)
        self.dense_block = nn.ModuleList(dense_block)

    def forward(self, features: list):
        """
        :param features:
            list containing feature descriptors at different spatial resolution
                0: [2N, C0, H//4, W//4]
                1: [2N, C1, H//8, W//8]
                2: [2N, C2, H//16, W//16]
        :return: feature descriptor at full resolution [2N,C,H//4,W//4]
        """

        features.reverse()
        output = self.bottle_neck(features[0])
        output = output[:, -(self.block_config * self.growth_rate):]  # take only the new features

        for i in range(len(features) - 1):
            hs = self.up[i](output, features[i + 1])  # scale up and concat
            output = self.dense_block[i](hs)  # denseblock
            if i == len(features) - 2:
                output = output[:, -self.hidden_dim:]
            else:
                output = output[:, -(self.block_config * self.growth_rate):]  # take only the new features

        return output


def build_tokenizer(args, layer_channel):
    block_config = 4
    growth_rate = 16
    return Tokenizer(block_config, args.channel_dim, layer_channel, growth_rate, use_instance_norm=args.instance_norm)
