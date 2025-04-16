# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
YOLO-NAS model interface.

Example:
    ```python
    from ultralytics import NAS

    model = NAS('yolo_nas_s')
    results = model.predict('ultralytics/assets/bus.jpg')
    ```
"""

from pathlib import Path

import torch

from ultralytics.engine.model import Model
from ultralytics.utils.torch_utils import model_info, smart_inference_mode

from .predict import NASPredictor
from .val import NASValidator


class NAS(Model):
    """
    YOLO NAS model for object detection.

    This class provides an interface for the YOLO-NAS models and extends the `Model` class from Ultralytics engine.
    It is designed to facilitate the task of object detection using pre-trained or custom-trained YOLO-NAS models.

    Example:
        ```python
        from ultralytics import NAS

        model = NAS('yolo_nas_s')
        results = model.predict('ultralytics/assets/bus.jpg')
        ```

    Attributes:
        model (str): Path to the pre-trained model or model name. Defaults to 'yolo_nas_s.pt'.

    Note:
        YOLO-NAS models only support pre-trained models. Do not provide YAML configuration files.
    """


    @smart_inference_mode()


    @property