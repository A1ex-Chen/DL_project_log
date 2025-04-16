# Taken from https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/peleenet.py

import torch
import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct


class PeleeBranch1(nn.Module):
    """
    PeleeNet branch type 1 block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of intermediate channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the second convolution.
    """




class PeleeBranch2(nn.Module):
    """
    PeleeNet branch type 2 block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of intermediate channels.
    """




class TwoStackDenseBlock(nn.Module):
    """
    PeleeNet dense block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bottleneck_size : int
        Bottleneck width.
    """

