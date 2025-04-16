# Code taken from: https://github.com/kuangliu/pytorch-cifar

import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """Grouped convolution block."""

    expansion = 2




class ResNeXt(nn.Module):

