import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.layers import ResnetBlockFC


class VelocityField(nn.Module):
    ''' Velocity Field network class.

    It maps input points and time values together with (optional) conditioned
    codes c and latent codes z to the respective motion vectors.

    Args:
        in_dim (int): input dimension of points concatenated with the time axis
        out_dim (int): output dimension of motion vectors
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): size of the hidden dimension
        leaky (bool): whether to use leaky ReLUs as activation
        n_blocks (int): number of ResNet-based blocks
    '''




