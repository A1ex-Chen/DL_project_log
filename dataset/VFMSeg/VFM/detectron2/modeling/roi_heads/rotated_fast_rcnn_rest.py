# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
import torch

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms_rotated
from detectron2.structures import Instances, RotatedBoxes, pairwise_iou_rotated
from detectron2.utils.events import get_event_storage

from ..box_regression import Box2BoxTransformRotated
from ..poolers import ROIPooler
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers
from .roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 5-d (dx, dy, dw, dh, da) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransformRotated`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted rotated box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth rotated box2box transform deltas
"""






class RotatedFastRCNNOutputLayers(FastRCNNOutputLayers):
    """
    Two linear layers for predicting Rotated Fast R-CNN outputs.
    """

    @classmethod



@ROI_HEADS_REGISTRY.register()
class RROIHeads(StandardROIHeads):
    """
    This class is used by Rotated Fast R-CNN to detect rotated boxes.
    For now, it only supports box predictions but not mask or keypoints.
    """

    @configurable

    @classmethod

    @torch.no_grad()