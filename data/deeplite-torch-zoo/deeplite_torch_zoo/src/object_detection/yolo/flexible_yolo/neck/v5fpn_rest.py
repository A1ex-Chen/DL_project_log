# Code credit: https://github.com/Bobo-y/flexible-yolov5
# The file is modified by Deeplite Inc. from the original implementation on Jan 4, 2023

from functools import partial

import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct as Conv, Concat
from deeplite_torch_zoo.src.dnn_blocks.yolov8.yolo_ultralytics_blocks import YOLOC3
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.neck.neck_utils import YOLO_SCALING_GAINS

from deeplite_torch_zoo.utils import LOGGER, make_divisible


class YOLOv5FPN(nn.Module):
    """
    YOLOv5 FPN module

         concat
    C3 --->   P3
    |          ^
    V   concat | up2
    C4 --->   P4
    |          ^
    V          | up2
    C5 --->    P5
    """




