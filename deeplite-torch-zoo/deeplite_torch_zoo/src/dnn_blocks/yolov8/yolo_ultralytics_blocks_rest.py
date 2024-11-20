# Ultralytics YOLO 🚀, AGPL-3.0 license

import functools

import numpy as np
import torch
import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct
from deeplite_torch_zoo.src.dnn_blocks.yolov7.yolo_blocks import (
    YOLOBottleneck,
    YOLOGhostBottleneck,
)
from deeplite_torch_zoo.src.dnn_blocks.yolov7.yolo_spp_blocks import YOLOSPP
from deeplite_torch_zoo.src.registries import EXPANDABLE_BLOCKS, VARIABLE_CHANNEL_BLOCKS


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOC3(nn.Module):
    # CSP Bottleneck with 3 convolutions



@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOC2(nn.Module):
    # CSP Bottleneck with 2 convolutions



@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOC2f(nn.Module):
    # CSP Bottleneck with 2 convolutions




@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOC1(nn.Module):
    # CSP Bottleneck with 1 convolution



@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOC3x(YOLOC3):
    # C3 module with cross-convolutions


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOC3Ghost(YOLOC3):
    # C3 module with GhostBottleneck()


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOC3TR(YOLOC3):
    # C3 module with TransformerBlock()


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOC3SPP(YOLOC3):
    # C3 module with SPP()


@VARIABLE_CHANNEL_BLOCKS.register()
class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)



@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOTransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
