# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List
import torch
from torch import nn
from torch.autograd.function import Function

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..poolers import ROIPooler
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers, fast_rcnn_inference
from .roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads


class _ScaleGradient(Function):
    @staticmethod

    @staticmethod


@ROI_HEADS_REGISTRY.register()
class CascadeROIHeads(StandardROIHeads):
    """
    The ROI heads that implement :paper:`Cascade R-CNN`.
    """

    @configurable

    @classmethod

    @classmethod



    @torch.no_grad()

