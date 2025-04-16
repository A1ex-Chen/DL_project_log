import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from functools import partial
from helpers import to_2tuple

from bitlinear import BitLinear

__all__ = [
    "Mlp",
    "Attention"
]

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    

class Attention(nn.Module):

