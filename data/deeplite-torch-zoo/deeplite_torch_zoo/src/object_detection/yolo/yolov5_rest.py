# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# The file is modified by Deeplite Inc. from the original implementation on Mar 21, 2023

import math

import torch
import torch.nn as nn

from deeplite_torch_zoo.src.object_detection.yolo.heads import (
    Detect, DetectV8, DetectX
)
from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct as Conv
from deeplite_torch_zoo.src.dnn_blocks.common import DWConv
from deeplite_torch_zoo.src.dnn_blocks.yolov7.repvgg_blocks import RepConv
from deeplite_torch_zoo.src.dnn_blocks.mobileone.mobileone_blocks import MobileOneBlock

from deeplite_torch_zoo.utils import (
    LOGGER,
    fuse_conv_and_bn,
    model_info,
    scale_img,
)


class DetectionModel(nn.Module):











