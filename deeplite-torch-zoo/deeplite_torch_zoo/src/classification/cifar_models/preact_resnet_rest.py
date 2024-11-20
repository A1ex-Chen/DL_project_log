# Code taken from: https://github.com/kuangliu/pytorch-cifar

import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1




class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4




class PreActResNet(nn.Module):

