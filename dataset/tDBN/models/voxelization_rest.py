import time
from enum import Enum
from functools import reduce

import numpy as np
import sparseconvnet as scn
import torch
from torch import nn
from torch.nn import functional as F

import torchplus
from torchplus import metrics
from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.ops.array_ops import gather_nd, scatter_nd
from torchplus.tools import change_default_args
from tDBN.core import box_torch_ops
import operator
import torch
import warnings
#from torch.nn.parallel.data_parallel import *

class VFELayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, name='vfe'):
        super(VFELayer, self).__init__()
        self.name = name
        self.units = int(out_channels / 2)
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1d(self.units)

    def forward(self, inputs):
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        voxel_count = inputs.shape[1]
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        pointwise = F.relu(x)
        # [K, T, units]

        aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        # [K, 1, units]
        repeated = aggregated.repeat(1, voxel_count, 1)

        concatenated = torch.cat([pointwise, repeated], dim=2)
        # [K, T, 2 * units]
        return concatenated





def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class VoxelFeatureExtractor(nn.Module):



class BinaryVoxel(nn.Module):





