from typing import Union, Tuple, List, Dict, Optional

import torch
import torchvision.models.detection
import torchvision.models.detection.transform
from torch import Tensor, nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import mobilenet_backbone, resnet_fpn_backbone
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import resize_boxes

from . import Algorithm
from ..backbone import Backbone
from ..backbone.mobilenet_v3_small import MobileNet_v3_Small
from ..backbone.mobilenet_v3_large import MobileNet_v3_Large
from ..backbone.resnet50 import ResNet50


class TorchFPN(Algorithm):




    class IdentityTransform(nn.Module):

