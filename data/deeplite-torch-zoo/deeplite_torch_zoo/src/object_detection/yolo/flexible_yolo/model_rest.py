# Code credit: https://github.com/Bobo-y/flexible-yolov5

import math
import yaml

from addict import Dict

import torch
from torch import nn

from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.backbone import (
    build_backbone,
)
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.neck import build_neck
from deeplite_torch_zoo.src.object_detection.yolo.heads import Detect, DetectV8
from deeplite_torch_zoo.src.object_detection.yolo.yolov5 import (
    Conv,
    DWConv,
    RepConv,
    DetectionModel,
    fuse_conv_and_bn,
)
from deeplite_torch_zoo.src.object_detection.yolo.config_parser import HEAD_NAME_MAP
from deeplite_torch_zoo.utils import initialize_weights, LOGGER


class FlexibleYOLO(DetectionModel):






