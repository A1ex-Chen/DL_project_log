# 2020.11.06-Changed for building GhostNetV2
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

# Creates a GhostNet Model as defined in:
# GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
# https://arxiv.org/abs/1911.11907
# Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models

# Taken from https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch
# The file is modified by Deeplite Inc. from the original implementation on Jan 18, 2023
# Code implementation refactoring

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplite_torch_zoo.src.registries import VARIABLE_CHANNEL_BLOCKS
from deeplite_torch_zoo.src.dnn_blocks.common import get_activation, ConvBnAct
from deeplite_torch_zoo.src.dnn_blocks.pytorchcv.cnn_attention import SELayer


class DFCModule(nn.Module):



class GhostModuleV2(nn.Module):



class GhostBottleneckV2(nn.Module):



@VARIABLE_CHANNEL_BLOCKS.register()
class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
