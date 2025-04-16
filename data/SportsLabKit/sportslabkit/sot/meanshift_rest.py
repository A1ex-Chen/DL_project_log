import cv2
import numpy as np

from sportslabkit.sot.base import SingleObjectTracker


class MeanShiftTracker(SingleObjectTracker):
    required_keys = ["box"]




    @property