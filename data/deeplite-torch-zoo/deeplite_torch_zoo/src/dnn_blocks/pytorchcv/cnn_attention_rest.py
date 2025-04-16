# Taken from: https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/common.py

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplite_torch_zoo.src.dnn_blocks.common import get_activation, round_channels


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    mid_channels : int or None, default None
        Number of middle channels.
    round_mid : bool, default False
        Whether to round middle channel number (make divisible by 8).
    use_conv : bool, default True
        Whether to convolutional layers instead of fully-connected ones.
    activation : function, or str, or nn.Module, default 'relu'
        Activation function after the first convolution.
    out_activation : function, or str, or nn.Module, default 'sigmoid'
        Activation function after the last convolution.
    """




SEWithNorm = partial(SELayer, norm_layer=nn.BatchNorm2d)


class CBAM(nn.Module):



class SpatialAttention(nn.Module):




class ChannelAttention(nn.Module):
