# Copyright (c) Facebook, Inc. and its affiliates.
import torch.nn as nn

from detectron2.modeling import ShapeSpec

__all__ = ["Backbone"]


class Backbone(nn.Module):
    """
    Abstract base class for network backbones.
    """



    @property
