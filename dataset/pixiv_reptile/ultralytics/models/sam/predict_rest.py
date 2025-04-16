# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Generate predictions using the Segment Anything Model (SAM).

SAM is an advanced image segmentation model offering features like promptable segmentation and zero-shot performance.
This module contains the implementation of the prediction logic and auxiliary utilities required to perform segmentation
using SAM. It forms an integral part of the Ultralytics framework and is designed for high-performance, real-time image
segmentation tasks.
"""

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.utils.torch_utils import select_device

from .amg import (
    batch_iterator,
    batched_mask_to_box,
    build_all_layer_point_grids,
    calculate_stability_score,
    generate_crop_boxes,
    is_box_near_crop_edge,
    remove_small_regions,
    uncrop_boxes_xyxy,
    uncrop_masks,
)
from .build import build_sam


class Predictor(BasePredictor):
    """
    Predictor class for the Segment Anything Model (SAM), extending BasePredictor.

    The class provides an interface for model inference tailored to image segmentation tasks.
    With advanced architecture and promptable segmentation capabilities, it facilitates flexible and real-time
    mask generation. The class is capable of working with various types of prompts such as bounding boxes,
    points, and low-resolution masks.

    Attributes:
        cfg (dict): Configuration dictionary specifying model and task-related parameters.
        overrides (dict): Dictionary containing values that override the default configuration.
        _callbacks (dict): Dictionary of user-defined callback functions to augment behavior.
        args (namespace): Namespace to hold command-line arguments or other operational variables.
        im (torch.Tensor): Preprocessed input image tensor.
        features (torch.Tensor): Extracted image features used for inference.
        prompts (dict): Collection of various prompt types, such as bounding boxes and points.
        segment_all (bool): Flag to control whether to segment all objects in the image or only specified ones.
    """













    @staticmethod