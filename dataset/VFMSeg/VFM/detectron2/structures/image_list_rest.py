# Copyright (c) Facebook, Inc. and its affiliates.
from __future__ import division
from typing import Any, List, Tuple
import torch
from torch import device
from torch.nn import functional as F

from detectron2.layers.wrappers import shapes_to_tensor


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size.
    The original sizes of each image is stored in `image_sizes`.

    Attributes:
        image_sizes (list[tuple[int, int]]): each tuple is (h, w).
            During tracing, it becomes list[Tensor] instead.
    """




    @torch.jit.unused

    @property

    @staticmethod