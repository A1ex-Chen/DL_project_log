# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.engine.results import Results
from ultralytics.models.fastsam.utils import bbox_iou
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class FastSAMPredictor(DetectionPredictor):
    """
    FastSAMPredictor is specialized for fast SAM (Segment Anything Model) segmentation prediction tasks in Ultralytics
    YOLO framework.

    This class extends the DetectionPredictor, customizing the prediction pipeline specifically for fast SAM.
    It adjusts post-processing steps to incorporate mask prediction and non-max suppression while optimizing
    for single-class segmentation.

    Attributes:
        cfg (dict): Configuration parameters for prediction.
        overrides (dict, optional): Optional parameter overrides for custom behavior.
        _callbacks (dict, optional): Optional list of callback functions to be invoked during prediction.
    """

