# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """


    @staticmethod


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""


    @staticmethod


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""




class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""




class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""




class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""




class v8DetectionLoss:
    """Criterion class for computing training losses."""






class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""



    @staticmethod



class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""



    @staticmethod



class v8ClassificationLoss:
    """Criterion class for computing training losses."""



class v8OBBLoss(v8DetectionLoss):





class E2EDetectLoss:
    """Criterion class for computing training losses."""

