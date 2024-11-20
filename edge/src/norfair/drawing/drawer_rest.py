from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from norfair.drawing.color import Color, ColorType
from norfair.tracker import Detection, TrackedObject

try:
    import cv2
except ImportError:
    from norfair.utils import DummyOpenCVImport

    cv2 = DummyOpenCVImport()


class Drawer:
    """
    Basic drawing functionality.

    This class encapsulates opencv drawing functions allowing for
    different backends to be implemented following the same interface.
    """

    @classmethod

    @classmethod

    @classmethod

    @classmethod

    @classmethod

    @classmethod


class Drawable:
    """
    Class to standardize Drawable objects like Detections and TrackedObjects

    Parameters
    ----------
    obj : Union[Detection, TrackedObject], optional
        A [Detection][norfair.tracker.Detection] or a [TrackedObject][norfair.tracker.TrackedObject]
        that will be used to initialized the drawable.
        If this parameter is passed, all other arguments are ignored
    points : np.ndarray, optional
        Points included in the drawable, shape is `(N_points, N_dimensions)`. Ignored if `obj` is passed
    id : Any, optional
        Id of this object. Ignored if `obj` is passed
    label : Any, optional
        Label specifying the class of the object. Ignored if `obj` is passed
    scores : np.ndarray, optional
        Confidence scores of each point, shape is `(N_points,)`. Ignored if `obj` is passed
    live_points : np.ndarray, optional
        Bolean array indicating which points are alive, shape is `(N_points,)`. Ignored if `obj` is passed

    Raises
    ------
    ValueError
        If obj is not an instance of the supported classes.
    """
