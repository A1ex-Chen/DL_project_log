import torch.nn as nn

from deeplite_torch_zoo.utils import initialize_weights

from deeplite_torch_zoo.src.object_detection.yolo.heads import Detect
from deeplite_torch_zoo.src.object_detection.yolo.config_parser import HEAD_NAME_MAP
from deeplite_torch_zoo.src.object_detection.yolo.anchors import ANCHOR_REGISTRY

from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.model import FlexibleYOLO
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.neck import build_neck
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.backbone.timm_wrapper import TimmWrapperBackbone



class TimmYOLO(FlexibleYOLO):