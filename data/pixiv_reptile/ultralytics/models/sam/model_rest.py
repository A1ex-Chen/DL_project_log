# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
SAM model interface.

This module provides an interface to the Segment Anything Model (SAM) from Ultralytics, designed for real-time image
segmentation tasks. The SAM model allows for promptable segmentation with unparalleled versatility in image analysis,
and has been trained on the SA-1B dataset. It features zero-shot performance capabilities, enabling it to adapt to new
image distributions and tasks without prior knowledge.

Key Features:
    - Promptable segmentation
    - Real-time performance
    - Zero-shot transfer capabilities
    - Trained on SA-1B dataset
"""

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.utils.torch_utils import model_info

from .build import build_sam
from .predict import Predictor


class SAM(Model):
    """
    SAM (Segment Anything Model) interface class.

    SAM is designed for promptable real-time image segmentation. It can be used with a variety of prompts such as
    bounding boxes, points, or labels. The model has capabilities for zero-shot performance and is trained on the SA-1B
    dataset.
    """






    @property