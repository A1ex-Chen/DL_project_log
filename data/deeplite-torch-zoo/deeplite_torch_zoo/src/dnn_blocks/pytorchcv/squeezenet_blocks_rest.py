# Taken from:
# - https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/squeezenet.py
# - https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/squeezenext.py

import torch
import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct, get_activation


class FireUnit(nn.Module):
    """
    SqueezeNet unit, so-called 'Fire' unit.
    Parameters:
    ----------
    c1 : int
        Number of input channels.
    c2 : int
        Number of output channels.
    e : float , default 1/8
        Number of internal channels for squeeze convolution blocks.
    act : string
        Activation function to be used
    residual : bool
        Whether use residual connection.
    """




class SqnxtUnit(nn.Module):
    """
    SqueezeNext unit.
    Original paper: 'SqueezeNext: Hardware-Aware Neural Network Design,'
    https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    c1 : int
        Number of input channels.
    c2 : int
        Number of output channels.
    s : int or tuple/list of 2 int
        Strides of the convolution.
    """

