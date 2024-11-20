#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

from typing import List

import numpy as np

from .base_tracker import TRACKER_HEADS_REGISTRY
from .vanilla_hungarian_bbox_iou_tracker import VanillaHungarianBBoxIOUTracker
from detectron2.config import configurable, CfgNode as CfgNode_


@TRACKER_HEADS_REGISTRY.register()
class IOUWeightedHungarianBBoxIOUTracker(VanillaHungarianBBoxIOUTracker):
    """
    A tracker using IoU as weight in Hungarian algorithm, also known
    as Munkres or Kuhn-Munkres algorithm
    """

    @configurable

    @classmethod
