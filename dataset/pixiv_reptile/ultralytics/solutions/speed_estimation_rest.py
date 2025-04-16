# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict
from time import time

import cv2
import numpy as np

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors


class SpeedEstimator:
    """A class to estimate the speed of objects in a real-time video stream based on their tracks."""









if __name__ == "__main__":
    names = {0: "person", 1: "car"}  # example class names
    speed_estimator = SpeedEstimator(names)