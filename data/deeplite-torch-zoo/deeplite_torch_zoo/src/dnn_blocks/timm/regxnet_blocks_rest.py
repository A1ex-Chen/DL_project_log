""" ReXNet

A PyTorch impl of `ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network` -
https://arxiv.org/abs/2007.00992

Adapted from original impl at https://github.com/clovaai/rexnet
Copyright (c) 2020-present NAVER Corp. MIT license

Changes for timm, feature extraction, and rounded channel variant hacked together by Ross Wightman
Copyright 2020 Ross Wightman
"""

# Modified from https://github1s.com/huggingface/pytorch-image-models/blob/HEAD/timm/models/rexnet.py
# The file is modified by Deeplite Inc. from the original implementation on Dec 22, 2022
# Code implementation refactoring

import torch
import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import (
    ConvBnAct,
    get_activation,
    round_channels,
)
from deeplite_torch_zoo.src.dnn_blocks.pytorchcv.cnn_attention import SEWithNorm


class RexNetBottleneck(nn.Module):
