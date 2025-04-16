#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

from typing import List

import numpy as np
from detectron2.structures import Instances
from detectron2.structures.boxes import pairwise_iou
from detectron2.tracking.utils import create_prediction_pairs, LARGE_COST_VALUE

from .base_tracker import TRACKER_HEADS_REGISTRY
from .hungarian_tracker import BaseHungarianTracker
from detectron2.config import configurable, CfgNode as CfgNode_


@TRACKER_HEADS_REGISTRY.register()
class VanillaHungarianBBoxIOUTracker(BaseHungarianTracker):
    """
    Hungarian algo based tracker using bbox iou as metric
    """

    @configurable

    @classmethod

