# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
from typing import Dict, List
import torch

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms_rotated, cat
from detectron2.structures import Instances, RotatedBoxes, pairwise_iou_rotated
from detectron2.utils.memory import retry_if_cuda_oom

from ..box_regression import Box2BoxTransformRotated
from .build import PROPOSAL_GENERATOR_REGISTRY
from .proposal_utils import _is_tracing
from .rpn import RPN

logger = logging.getLogger(__name__)




@PROPOSAL_GENERATOR_REGISTRY.register()
class RRPN(RPN):
    """
    Rotated Region Proposal Network described in :paper:`RRPN`.
    """

    @configurable

    @classmethod

    @torch.no_grad()

    @torch.no_grad()