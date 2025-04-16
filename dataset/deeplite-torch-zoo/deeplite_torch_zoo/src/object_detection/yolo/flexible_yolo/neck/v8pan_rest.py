# Code credit: https://github.com/Bobo-y/flexible-yolov5

from functools import partial

import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct as Conv, Concat
from deeplite_torch_zoo.src.dnn_blocks.yolov8.yolo_ultralytics_blocks import YOLOC2f
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.neck.neck_utils import YOLO_SCALING_GAINS

from deeplite_torch_zoo.utils import LOGGER, make_divisible


class YOLOv8PAN(nn.Module):
    """
    YOLOv8 PAN module
    P3 --->  PP3
    ^         |
    | concat  V
    P4 --->  PP4
    ^         |
    | concat  V
    P5 --->  PP5
    """




