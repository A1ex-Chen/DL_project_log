# Modified from https://github.com/moskomule/senet.pytorch

import torch.nn as nn
from torch import Tensor

from deeplite_torch_zoo.src.dnn_blocks.common import (
    ConvBnAct,
    DWConv,
    get_activation,
    round_channels,
)
from deeplite_torch_zoo.src.dnn_blocks.pytorchcv.cnn_attention import SELayer
from deeplite_torch_zoo.src.dnn_blocks.ghostnetv2.ghostnet_blocks import GhostConv


class ResNetBottleneck(nn.Module):
    # https://github.com/moskomule/senet.pytorch/blob/master/senet/se_resnet.py



class ResNetBasicBlock(nn.Module):
    # https://github.com/moskomule/senet.pytorch/blob/master/senet/se_resnet.py



class ResNeXtBottleneck(ResNetBottleneck):


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck as described in https://github.com/huawei-noah/ghostnet
