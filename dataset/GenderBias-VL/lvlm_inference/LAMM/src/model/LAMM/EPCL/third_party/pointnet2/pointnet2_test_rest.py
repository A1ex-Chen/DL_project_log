# Copyright (c) Facebook, Inc. and its affiliates.

""" Testing customized ops. """

import torch
from torch.autograd import gradcheck
import numpy as np

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import pointnet2_utils



    assert gradcheck(interpolate_func, feats, atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    test_interpolation_grad()