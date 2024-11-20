# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import List, Optional, Tuple
import torch
from fvcore.nn import sigmoid_focal_loss_jit
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, batched_nms
from detectron2.modeling.matcher import Matcher
from detectron2.structures import Boxes, ImageList, Instances, pairwise_point_box_distance
from detectron2.utils.events import get_event_storage

from ..anchor_generator import DefaultAnchorGenerator
from ..backbone import Backbone
from ..box_regression import Box2BoxTransformLinear, _dense_box_regression_loss
from .dense_detector import DenseDetector
from .retinanet import RetinaNetHead

__all__ = ["FCOS"]

logger = logging.getLogger(__name__)


class FCOS(DenseDetector):
    """
    Implement FCOS in :paper:`fcos`.
    """



    @torch.no_grad()

    @torch.no_grad()






class FCOSHead(RetinaNetHead):
    """
    The head used in :paper:`fcos`. It adds an additional centerness
    prediction branch on top of :class:`RetinaNetHead`.
    """

