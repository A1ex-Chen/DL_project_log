import torch
import torch.nn as nn


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''




class CResnetBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.

    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks 
    '''




class ResnetBlockConv1d(nn.Module):
    ''' 1D-Convolutional ResNet block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''




# Utility modules
class AffineLayer(nn.Module):
    ''' Affine layer class.

    Args:
        c_dim (tensor): dimension of latent conditioned code c
        dim (int): input dimension
    '''





class CBatchNorm1d(nn.Module):
    ''' Conditional batch normalization layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''





class CBatchNorm1d_legacy(nn.Module):
    ''' Conditional batch normalization legacy layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''


