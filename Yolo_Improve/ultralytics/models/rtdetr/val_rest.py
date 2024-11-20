# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops

__all__ = ("RTDETRValidator",)  # tuple or list


class RTDETRDataset(YOLODataset):
    """
    Real-Time DEtection and TRacking (RT-DETR) dataset class extending the base YOLODataset class.

    This specialized dataset class is designed for use with the RT-DETR object detection model and is optimized for
    real-time detection and tracking tasks.
    """


    # NOTE: add stretch version load_image for RTDETR mosaic



class RTDETRValidator(DetectionValidator):
    """
    RTDETRValidator extends the DetectionValidator class to provide validation capabilities specifically tailored for
    the RT-DETR (Real-Time DETR) object detection model.

    The class allows building of an RTDETR-specific dataset for validation, applies Non-maximum suppression for
    post-processing, and updates evaluation metrics accordingly.

    Example:
        ```python
        from ultralytics.models.rtdetr import RTDETRValidator

        args = dict(model='rtdetr-l.pt', data='coco8.yaml')
        validator = RTDETRValidator(args=args)
        validator()
        ```

    Note:
        For further details on the attributes and methods, refer to the parent DetectionValidator class.
    """



