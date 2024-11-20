# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import abc
from itertools import repeat
from numbers import Number
from typing import List

import numpy as np

from .ops import ltwh2xywh, ltwh2xyxy, xywh2ltwh, xywh2xyxy, xyxy2ltwh, xyxy2xywh



    return parse


to_2tuple = _ntuple(2)
to_4tuple = _ntuple(4)

# `xyxy` means left top and right bottom
# `xywh` means center x, center y and width, height(YOLO format)
# `ltwh` means left top and width, height(COCO format)
_formats = ["xyxy", "xywh", "ltwh"]

__all__ = ("Bboxes",)  # tuple or list


class Bboxes:
    """
    A class for handling bounding boxes.

    The class supports various bounding box formats like 'xyxy', 'xywh', and 'ltwh'.
    Bounding box data should be provided in numpy arrays.

    Attributes:
        bboxes (numpy.ndarray): The bounding boxes stored in a 2D numpy array.
        format (str): The format of the bounding boxes ('xyxy', 'xywh', or 'ltwh').

    Note:
        This class does not handle normalization or denormalization of bounding boxes.
    """

        # self.normalized = normalized



    # def denormalize(self, w, h):
    #    if not self.normalized:
    #         return
    #     assert (self.bboxes <= 1.0).all()
    #     self.bboxes[:, 0::2] *= w
    #     self.bboxes[:, 1::2] *= h
    #     self.normalized = False
    #
    # def normalize(self, w, h):
    #     if self.normalized:
    #         return
    #     assert (self.bboxes > 1.0).any()
    #     self.bboxes[:, 0::2] /= w
    #     self.bboxes[:, 1::2] /= h
    #     self.normalized = True




    @classmethod



class Instances:
    """
    Container for bounding boxes, segments, and keypoints of detected objects in an image.

    Attributes:
        _bboxes (Bboxes): Internal object for handling bounding box operations.
        keypoints (ndarray): keypoints(x, y, visible) with shape [N, 17, 3]. Default is None.
        normalized (bool): Flag indicating whether the bounding box coordinates are normalized.
        segments (ndarray): Segments array with shape [N, 1000, 2] after resampling.

    Args:
        bboxes (ndarray): An array of bounding boxes with shape [N, 4].
        segments (list | ndarray, optional): A list or array of object segments. Default is None.
        keypoints (ndarray, optional): An array of keypoints with shape [N, 17, 3]. Default is None.
        bbox_format (str, optional): The format of bounding boxes ('xywh' or 'xyxy'). Default is 'xywh'.
        normalized (bool, optional): Whether the bounding box coordinates are normalized. Default is True.

    Examples:
        ```python
        # Create an Instances object
        instances = Instances(
            bboxes=np.array([[10, 10, 30, 30], [20, 20, 40, 40]]),
            segments=[np.array([[5, 5], [10, 10]]), np.array([[15, 15], [20, 20]])],
            keypoints=np.array([[[5, 5, 1], [10, 10, 1]], [[15, 15, 1], [20, 20, 1]]])
        )
        ```

    Note:
        The bounding box format is either 'xywh' or 'xyxy', and is determined by the `bbox_format` argument.
        This class does not perform input validation, and it assumes the inputs are well-formed.
    """



    @property












    @classmethod

    @property