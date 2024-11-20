import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.layers import ResnetBlockFC




class Encoder(nn.Module):
    ''' Latent encoder class.

    It encodes input points together with their occupancy values and an
    (optional) conditioned latent code c to mean and standard deviations of
    the posterior distribution.

    Args:
        dim (int): dimension of input points
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): dimension of hidden size
        leaky (bool): whether to use leaky ReLUs as activation 

    '''




class PointNet(nn.Module):
    ''' Latent PointNet-based encoder class.

    It maps the inputs together with an (optional) conditioned code c
    to means and standard deviations.

    Args:
        dim (int): dimension of input points
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_dim (int): dimension of hidden size
        n_blocks (int): number of ResNet-based blocks
    '''

