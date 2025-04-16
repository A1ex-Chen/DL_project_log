# Copyright (c) Facebook, Inc. and its affiliates.

from copy import deepcopy
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from .batch_norm import get_norm
from .blocks import DepthwiseSeparableConv2d
from .wrappers import Conv2d


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    """

