# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import numpy as np
import os
import torch
from pycocotools.cocoeval import COCOeval, maskUtils

from detectron2.structures import BoxMode, RotatedBoxes, pairwise_iou_rotated
from detectron2.utils.file_io import PathManager

from .coco_evaluation import COCOEvaluator


class RotatedCOCOeval(COCOeval):
    @staticmethod

    @staticmethod




class RotatedCOCOEvaluator(COCOEvaluator):
    """
    Evaluate object proposal/instance detection outputs using COCO-like metrics and APIs,
    with rotated boxes support.
    Note: this uses IOU only and does not consider angle differences.
    """



