# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np




class RandomCuboid(object):
    """
    RandomCuboid augmentation from DepthContrast [https://arxiv.org/abs/2101.02691]
    We slightly modify this operation to account for object detection.
    This augmentation randomly crops a cuboid from the input and
    ensures that the cropped cuboid contains at least one bounding box
    """

