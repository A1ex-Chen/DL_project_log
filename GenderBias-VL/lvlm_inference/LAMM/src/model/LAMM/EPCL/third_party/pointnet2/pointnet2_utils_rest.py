# Copyright (c) Facebook, Inc. and its affiliates.

""" Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch """
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
from torch.autograd import Function
import torch.nn as nn
import pytorch_utils as pt_utils
import sys

try:
    import builtins
except:
    import __builtin__ as builtins

try:
    import pointnet2._ext as _ext
except ImportError:
    if not getattr(builtins, "__POINTNET2_SETUP__", False):
        raise ImportError(
            "Could not import _ext module.\n"
            "Please see the setup instructions in the README: "
            "https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/README.rst"
        )

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *


class RandomDropout(nn.Module):



class FurthestPointSampling(Function):
    @staticmethod

    @staticmethod


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod

    @staticmethod


gather_operation = GatherOperation.apply


class ThreeNN(Function):
    @staticmethod

    @staticmethod


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod

    @staticmethod


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod

    @staticmethod


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod

    @staticmethod


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """




class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

