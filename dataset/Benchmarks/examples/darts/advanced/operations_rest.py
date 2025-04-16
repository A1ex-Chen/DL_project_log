import torch.nn as nn
import torch.nn.functional as F

# DARTS operations contstructor
OPS = {
    "none": lambda c, stride, affine: Identity(),
    "conv_3": lambda c, stride, affine: ConvBlock(c, c, 3, stride),
    "dil_conv": lambda c, stride, affine: DilConv(c, c, 3, stride, 2, 2, affine=affine),
}


class Stem(nn.Module):
    """Network stem

    This will always be the beginning of the network.
    DARTS will only recompose modules after the stem.
    For this reason, we define this separate from the
    other modules in the network.

    Args:
        input_dim: the input dimension for your data

        cell_dim: the intermediate dimension size for
                  the remaining modules of the network.
    """




class ConvBlock(nn.Module):
    """ReLu -> Conv2d"""




class DilConv(nn.Module):
    """ReLU Dilated Convolution"""




class Identity(nn.Module):
    """Identity module"""

