# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import fvcore.nn.weight_init as weight_init
from torch import nn

from .batch_norm import FrozenBatchNorm2d, get_norm
from .wrappers import Conv2d


"""
CNN building blocks.
"""


class CNNBlockBase(nn.Module):
    """
    A CNN block is assumed to have input channels, output channels and a stride.
    The input and output of `forward()` method must be NCHW tensors.
    The method can perform arbitrary computation but must match the given
    channels and stride specification.

    Attribute:
        in_channels (int):
        out_channels (int):
        stride (int):
    """




class DepthwiseSeparableConv2d(nn.Module):
    """
    A kxk depthwise convolution + a 1x1 convolution.

    In :paper:`xception`, norm & activation are applied on the second conv.
    :paper:`mobilenet` uses norm & activation on both convs.
    """

