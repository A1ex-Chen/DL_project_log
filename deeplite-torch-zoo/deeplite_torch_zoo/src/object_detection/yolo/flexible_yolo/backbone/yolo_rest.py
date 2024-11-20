from functools import partial

import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct as Conv
from deeplite_torch_zoo.src.dnn_blocks.yolov8.yolo_ultralytics_blocks import (
    YOLOC3,
    YOLOC2f,
)
from deeplite_torch_zoo.src.dnn_blocks.yolov7.yolo_spp_blocks import YOLOSPPF
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.neck.neck_utils import (
    YOLO_SCALING_GAINS,
)

from deeplite_torch_zoo.utils import LOGGER, make_divisible


class YOLOv5Backbone(nn.Module):






class YOLOv8Backbone(YOLOv5Backbone):