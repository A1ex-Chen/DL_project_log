# ------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# To view a copy of this license, visit
# https://github.com/facebookresearch/detectron2/blob/main/LICENSE
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

import itertools
import json
import os
from collections import OrderedDict
from detectron2.evaluation import COCOEvaluator as _COCOEvaluator
from detectron2.evaluation import COCOPanopticEvaluator as _COCOPanopticEvaluator
from detectron2.evaluation import SemSegEvaluator as _SemSegEvaluator
from detectron2.evaluation.coco_evaluation import _evaluate_predictions_on_coco
from detectron2.utils.file_io import PathManager
from tabulate import tabulate


class COCOEvaluator(_COCOEvaluator):



class COCOPanopticEvaluator(_COCOPanopticEvaluator):



class SemSegEvaluator(_SemSegEvaluator):



# Copied from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/evaluation/instance_evaluation.py  # noqa
# modified from COCOEvaluator for instance segmentation
class InstanceSegEvaluator(COCOEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """
