# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Image augmentation functions
"""

import math
import random

import cv2
import numpy as np

from ..augmentations import box_candidates
from ..general import resample_segments, segment2box



