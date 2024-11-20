# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod
import torch.nn as nn

from detectron2.layers import ShapeSpec

__all__ = ["Backbone"]


class Backbone(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for network backbones.
    """


    @abstractmethod

    @property
