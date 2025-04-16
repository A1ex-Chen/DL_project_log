# Taken from:
# - https://github.com/WongKinYiu/yolov7/blob/HEAD/models/common.py
# The file is modified by Deeplite Inc. from the original implementation on Feb 23, 2023
# Refactoring block implementation

import functools

import numpy as np

import torch
import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import (
    ConvBnAct,
    DWConv,
    get_activation,
    round_channels,
    ACT_TYPE_MAP,
)
from deeplite_torch_zoo.src.dnn_blocks.resnet.resnet_blocks import (
    GhostBottleneck,
    ResNetBottleneck,
)
from deeplite_torch_zoo.src.dnn_blocks.ghostnetv2.ghostnet_blocks import GhostConv
from deeplite_torch_zoo.src.registries import EXPANDABLE_BLOCKS, VARIABLE_CHANNEL_BLOCKS


@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOBottleneck(nn.Module):
    # Ultralytics bottleneck (2 convs)



@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOBottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks



@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOBottleneckCSP2(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks



@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOVoVCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks



@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOGhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet



@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOBottleneckCSPF(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks



@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOBottleneckCSPL(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    # modified by @ivan-lazarevich to have c2 out channels



@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOBottleneckCSPLG(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    # modified by @lzrvch to have c2 out channels



@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class BottleneckCSPA(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks



@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class BottleneckCSPB(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks



@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class BottleneckCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks



@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class ResCSPA(BottleneckCSPA):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class ResCSPB(BottleneckCSPB):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class ResCSPC(BottleneckCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class GhostCSPA(BottleneckCSPA):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class GhostCSPB(BottleneckCSPB):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class GhostCSPC(BottleneckCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class ResXCSPA(ResCSPA):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class ResXCSPB(ResCSPB):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class ResXCSPC(ResCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks


@VARIABLE_CHANNEL_BLOCKS.register()
class Stem(nn.Module):
    # Stem



@VARIABLE_CHANNEL_BLOCKS.register()
class DownC(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP



@VARIABLE_CHANNEL_BLOCKS.register()
class MixConv2d(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595



@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOCrossConv(nn.Module):
    # Ultralytics Cross Convolution Downsample



@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOC4(nn.Module):
    # CSP Bottleneck with 4 convolutions aka old C3



class RobustConv(nn.Module):
    # Robust convolution (use high kernel size 7-11 for: downsampling and other layers). Train for 300 - 450 epochs.



class RobustConv2(nn.Module):
    # Robust convolution 2 (use [32, 5, 2] or [32, 7, 4] or [32, 11, 8] for one of the paths in CSP).
