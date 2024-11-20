# Copyright (c) Facebook, Inc. and its affiliates.

""" Utility functions for processing point clouds.

Author: Charles R. Qi and Or Litany
"""

import os
import sys
import torch

# Point cloud IO
import numpy as np


MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# ----------------------------------------
# Point Cloud Sampling
# ----------------------------------------






class RandomCuboid(object):
    """
    RandomCuboid augmentation from DepthContrast [https://arxiv.org/abs/2101.02691]
    We slightly modify this operation to account for object detection.
    This augmentation randomly crops a cuboid from the input and
    ensures that the cropped cuboid contains at least one bounding box
    """

