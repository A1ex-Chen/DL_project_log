#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
import copy

import numpy as np
import torch
from detectron2.structures import Boxes, Instances

from .base_tracker import BaseTracker
from scipy.optimize import linear_sum_assignment
from ..config.config import CfgNode as CfgNode_
from typing import Dict
from detectron2.config import configurable


class BaseHungarianTracker(BaseTracker):
    """
    A base class for all Hungarian trackers
    """

    @configurable

    @classmethod





