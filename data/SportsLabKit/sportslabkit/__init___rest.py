import logging
import warnings
from contextlib import contextmanager


@contextmanager


# load vidgear first with all_logging_disabled
with all_logging_disabled(), warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from vidgear.gears import CamGear, WriteGear

from sportslabkit.camera import Camera
from sportslabkit.dataframe import BBoxDataFrame, CoordinatesDataFrame
from sportslabkit.io import load_df
from sportslabkit.types.tracklet import Tracklet

import sportslabkit.datasets as datasets
import sportslabkit.detection_model
import sportslabkit.image_model
import sportslabkit.motion_model
import sportslabkit.calibration_model

__all__ = [
    "Camera",
    "Tracklet",
    "BBoxDataFrame",
    "CoordinatesDataFrame",
    "load_df",
    "CamGear",
    "WriteGear",
    "datasets",
]