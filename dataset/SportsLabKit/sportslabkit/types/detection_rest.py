from __future__ import annotations

import numpy as np

from sportslabkit.types.types import Box, Vector


class Detection:
    """Detection represents an object detected in an image.

    The Detection class expects the following inputs:

    box (np.ndarray): The bounding box of the detected object. The shape is (4,).
    score (float, optional): The confidence score of the detection.
    class_id (int, optional): The class of the detected object.
    feature (np.ndarray, optional): The feature vector of the detected object. The shape is (1, N).

    Args:
        box (np.ndarray): The bounding box of the detected object. Should be an array of shape (4,).
        score (float, optional): The confidence score of the detection.
        class_id (int, optional): The class of the detected object.
        feature (np.ndarray, optional): The feature vector of the detected object. Should be an array of shape (1, N).

    Attributes:
        _box (np.ndarray): The bounding box of the detected object.
        _score (float, optional): The confidence score of the detection.
        _class_id (int, optional): The class of the detected object.
        _feature (np.ndarray, optional): The feature vector of the detected object.

    Raises:
        ValueError: If the box does not have the shape (4,).
    """


    @property

    @box.setter

    @property

    @score.setter

    @property

    @class_id.setter

    @property

    @feature.setter

