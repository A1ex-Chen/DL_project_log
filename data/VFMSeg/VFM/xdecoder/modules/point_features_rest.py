# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch.nn import functional as F

from detectron2.layers import cat, shapes_to_tensor
from detectron2.structures import BitMasks, Boxes

# from ..layers import cat, shapes_to_tensor
# from ..structures import BitMasks, Boxes

"""
Shape shorthand in this module:

    N: minibatch dimension size, i.e. the number of RoIs for instance segmenation or the
        number of images for semantic segmenation.
    R: number of ROIs, combined over all images, in the minibatch
    P: number of points
"""













