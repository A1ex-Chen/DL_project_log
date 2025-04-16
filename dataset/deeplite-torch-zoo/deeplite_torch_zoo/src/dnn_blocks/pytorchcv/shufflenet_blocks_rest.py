# Taken from https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/shufflenet.py

import torch
import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct, DWConv, get_activation
from deeplite_torch_zoo.src.dnn_blocks.pytorchcv.common import ChannelShuffle


class ShuffleUnit(nn.Module):
    """
    ShuffleNet unit.
    Original paper: 'ShuffleNet: An Extremely Efficient Convolutional Neural Network
    for Mobile Devices,https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    c1 : int
        Number of input channels.
    c2 : int
        Number of output channels.
    g : int
        Number of groups in convolution layers.
    downsample : bool
        Whether do downsample.
    ignore_group : bool
        Whether ignore group value in the first convolution layer.
    """

