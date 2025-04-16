import torch
import torch.nn as nn
from lib.layers import ResnetBlockFC




class SimplePointnet(nn.Module):
    ''' PointNet-based encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''




class ResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''




class ResnetPointnet2Stream(nn.Module):
    ''' ResnetPointNet-based encoder network with two streams.

    The input point clouds are encoded with the same ResNet PointNet
    (shared weights) and the output codes are concatenated.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''




class TemporalResnetPointnet(nn.Module):
    ''' Temporal PointNet-based encoder network.

    The input point clouds are concatenated along the hidden dimension,
    e.g. for a sequence of length L, the dimension becomes 3xL = 51.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        use_only_first_pcl (bool): whether to use only the first point cloud
    '''

