# Ultralytics YOLO 🚀, AGPL-3.0 license

import getpass
from typing import List

import cv2
import numpy as np

from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER as logger
from ultralytics.utils import SETTINGS
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.plotting import plot_images









