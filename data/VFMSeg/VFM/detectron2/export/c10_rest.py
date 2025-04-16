# Copyright (c) Facebook, Inc. and its affiliates.

import math
import torch
import torch.nn.functional as F

from detectron2.layers import cat
from detectron2.layers.roi_align_rotated import ROIAlignRotated
from detectron2.modeling import poolers
from detectron2.modeling.proposal_generator import rpn
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference
from detectron2.structures import Boxes, ImageList, Instances, Keypoints

from .shared import alias, to_device


"""
This file contains caffe2-compatible implementation of several detectron2 components.
"""


class Caffe2Boxes(Boxes):
    """
    Representing a list of detectron2.structures.Boxes from minibatch, each box
    is represented by a 5d vector (batch index + 4 coordinates), or a 6d vector
    (batch index + 5 coordinates) for RotatedBoxes.
    """



# TODO clean up this class, maybe just extend Instances
class InstancesList(object):
    """
    Tensor representation of a list of Instances object for a batch of images.

    When dealing with a batch of images with Caffe2 ops, a list of bboxes
    (instances) are usually represented by single Tensor with size
    (sigma(Ni), 5) or (sigma(Ni), 4) plus a batch split Tensor. This class is
    for providing common functions to convert between these two representations.
    """









    @staticmethod


class Caffe2Compatible(object):
    """
    A model can inherit this class to indicate that it can be traced and deployed with caffe2.
    """



    tensor_mode = property(_get_tensor_mode, _set_tensor_mode)
    """
    If true, the model expects C2-style tensor only inputs/outputs format.
    """


class Caffe2RPN(Caffe2Compatible, rpn.RPN):


    @staticmethod


class Caffe2ROIPooler(Caffe2Compatible, poolers.ROIPooler):
    @staticmethod



class Caffe2FastRCNNOutputsInference:



class Caffe2MaskRCNNInference:


class Caffe2KeypointRCNNInference:
