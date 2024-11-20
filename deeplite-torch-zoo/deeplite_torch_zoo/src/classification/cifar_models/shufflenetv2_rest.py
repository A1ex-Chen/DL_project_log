# Code taken from: https://github.com/kuangliu/pytorch-cifar

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShuffleBlock(nn.Module):



class SplitBlock(nn.Module):



class BasicBlock(nn.Module):



class DownBlock(nn.Module):



class ShuffleNetV2(nn.Module):




configs = {
    0.5: {"out_channels": (48, 96, 192, 1024), "num_blocks": (3, 7, 3)},
    1: {"out_channels": (116, 232, 464, 1024), "num_blocks": (3, 7, 3)},
    1.5: {"out_channels": (176, 352, 704, 1024), "num_blocks": (3, 7, 3)},
    2: {"out_channels": (224, 488, 976, 2048), "num_blocks": (3, 7, 3)},
}