"""
CNN NLP operations closely modeled after the original paper's vision task.
"""

import torch
import torch.nn as nn
from darts.api import Model

OPS = {
    "none": lambda c, stride, affine: Zero(stride),
    "avg_pool_3": lambda c, stride, affine: nn.AvgPool1d(
        3, stride=stride, padding=1, count_include_pad=False
    ),
    "max_pool_3": lambda c, stride, affine: nn.MaxPool1d(3, stride=stride, padding=1),
    "skip_connect": lambda c, stride, affine: Identity()
    if stride == 1
    else FactorizedReduce(c, c, affine=affine),
    "sep_conv_3": lambda c, stride, affine: SepConv(c, c, 3, stride, 1, affine=affine),
    "sep_conv_5": lambda c, stride, affine: SepConv(c, c, 5, stride, 2, affine=affine),
    "sep_conv_7": lambda c, stride, affine: SepConv(c, c, 7, stride, 3, affine=affine),
    "dil_conv_3": lambda c, stride, affine: DilConv(
        c, c, 3, stride, 2, 2, affine=affine
    ),
    "dil_conv_5": lambda c, stride, affine: DilConv(
        c, c, 5, stride, 4, 2, affine=affine
    ),
    "convblock_7": lambda c, stride, affine: ConvBlock(
        c, c, 7, stride, 3, affine=affine
    ),
}


class ConvBlock(Model):
    """ReLu -> Conv1d -> BatchNorm"""




class DilConv(Model):
    """ReLU Dilated Convolution"""




class FactorizedReduce(Model):
    """Reduce the feature maps by half, maintaining number of channels

    Example
    -------
    x: torch.Size([2, 10, 12])
    out: [batch_size, c_out, d//2]
    out: torch.Size([2, 10, 6])
    """




class Identity(Model):



class SepConv(Model):
    """Separable Convolution Block"""




class Zero(nn.Module):
    """Zero tensor by stride"""

