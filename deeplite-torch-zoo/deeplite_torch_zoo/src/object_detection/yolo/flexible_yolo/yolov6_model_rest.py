import torch.nn as nn

from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.model import FlexibleYOLO
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.yolov6 import build_network, build_network_lite
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.yolov6.config import Config
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.yolov6.layers.common import (
    RepVGGBlock,
)

from deeplite_torch_zoo.src.object_detection.yolo.heads import Detect
from deeplite_torch_zoo.src.object_detection.yolo.anchors import ANCHOR_REGISTRY
from deeplite_torch_zoo.src.object_detection.yolo.yolov5 import (
    Conv,
    DWConv,
    RepConv,
    fuse_conv_and_bn,
)
from deeplite_torch_zoo.src.object_detection.yolo.config_parser import HEAD_NAME_MAP

from deeplite_torch_zoo.utils import initialize_weights, LOGGER


class YOLOv6(FlexibleYOLO):


