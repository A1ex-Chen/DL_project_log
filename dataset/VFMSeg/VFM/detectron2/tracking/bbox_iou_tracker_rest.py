#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
import copy
from typing import List

import numpy as np
import torch
from detectron2.config import configurable
from detectron2.structures import Boxes, Instances
from detectron2.structures.boxes import pairwise_iou

from ..config.config import CfgNode as CfgNode_
from .base_tracker import BaseTracker, TRACKER_HEADS_REGISTRY


@TRACKER_HEADS_REGISTRY.register()
class BBoxIOUTracker(BaseTracker):
    """
    A bounding box tracker to assign ID based on IoU between current and previous instances
    """
    @configurable

    @classmethod





