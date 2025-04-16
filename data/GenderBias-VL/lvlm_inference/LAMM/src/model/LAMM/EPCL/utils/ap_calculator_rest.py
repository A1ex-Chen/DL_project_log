# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Helper functions and class to calculate Average Precisions for 3D object detection.
"""
import logging
import os
import sys
from collections import OrderedDict

import numpy as np
import scipy.special as scipy_special
import torch

from utils.box_util import (
    extract_pc_in_box3d,
    flip_axis_to_camera_np,
    get_3d_box,
    get_3d_box_batch,
)
from utils.eval_det import eval_det_multiprocessing, get_iou_obb
from utils.nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls






# This is exactly the same as VoteNet so that we can compare evaluations.




class APCalculator(object):
    """Calculating Average Precision"""









