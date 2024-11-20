import torch.nn as nn
# import torch.nn.functional as F
from torchvision import models
from lib.common import normalize_imagenet


class Resnet18(nn.Module):
    r''' ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''




class ConvEncoder3D(nn.Module):
    r''' Simple convolutional conditioning network.

    It consists of 6 convolutional layers, each downsampling the input by a
    factor of 2, and a final fully-connected layer projecting the output to
    c_dim dimensions.
    '''

