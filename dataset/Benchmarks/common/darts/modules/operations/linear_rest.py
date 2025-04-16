"""
Linear operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from darts.api import Model

OPS = {
    "none": lambda c, stride, affice: Zero(),
    "skip_connect": lambda c, stride, affine: Identity(),
    "linear_block": lambda c, stride, affine: LinearBlock(c, c, affine=affine),
    "linear_conv": lambda c, stride, affine: LinearConv(c, c, 1),
    "linear_drop": lambda c, stride, affine: LinearDrop(c, c, 1),
    "encoder": lambda c, stride, affine: Encoder(c, c, 1),
}


class LinearBlock(Model):
    """Linear block consisting of two fully connected layers

    Example
    -------
    x: torch.Size([2, 10, 12])
    out: [batch_size, c_out, d//2]
    out: torch.Size([2, 10, 6])
    """




class LinearDrop(Model):
    """Linear block with dropout"""




class Encoder(Model):
    """Linear encoder"""




class LinearConv(Model):
    """Linear => Conv => Linear"""




class Identity(Model):



class Zero(nn.Module):
    """Zero tensor by stride"""

