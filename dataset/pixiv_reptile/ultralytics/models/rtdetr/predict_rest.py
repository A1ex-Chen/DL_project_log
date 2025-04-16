# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class RTDETRPredictor(BasePredictor):
    """
    RT-DETR (Real-Time Detection Transformer) Predictor extending the BasePredictor class for making predictions using
    Baidu's RT-DETR model.

    This class leverages the power of Vision Transformers to provide real-time object detection while maintaining
    high accuracy. It supports key features like efficient hybrid encoding and IoU-aware query selection.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.rtdetr import RTDETRPredictor

        args = dict(model='rtdetr-l.pt', source=ASSETS)
        predictor = RTDETRPredictor(overrides=args)
        predictor.predict_cli()
        ```

    Attributes:
        imgsz (int): Image size for inference (must be square and scale-filled).
        args (dict): Argument overrides for the predictor.
    """

