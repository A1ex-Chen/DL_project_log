# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
import numpy as np
from .utils.pc_util import shift_scale_points


class PositionEmbeddingCoordsSine(nn.Module):



