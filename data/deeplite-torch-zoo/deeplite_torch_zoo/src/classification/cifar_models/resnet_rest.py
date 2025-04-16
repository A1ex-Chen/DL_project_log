# Code taken from: https://github.com/kuangliu/pytorch-cifar

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1




class Bottleneck(nn.Module):
    expansion = 4




class ResNet(nn.Module):

