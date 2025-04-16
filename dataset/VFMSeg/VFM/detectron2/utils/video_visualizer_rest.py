# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import pycocotools.mask as mask_util

from detectron2.utils.visualizer import (
    ColorMode,
    Visualizer,
    _create_text_labels,
    _PanopticPrediction,
)

from .colormap import random_color, random_colors
from detectron2.structures import Instances
from typing import List


class _DetectedInstance:
    """
    Used to store data about detected objects in video frame,
    in order to transfer color to objects in the future frames.

    Attributes:
        label (int):
        bbox (tuple[float]):
        mask_rle (dict):
        color (tuple[float]): RGB colors in range (0, 1)
        ttl (int): time-to-live for the instance. For example, if ttl=2,
            the instance color can be transferred to objects in the next two frames.
    """

    __slots__ = ["label", "bbox", "mask_rle", "color", "ttl"]



class VideoVisualizer:




