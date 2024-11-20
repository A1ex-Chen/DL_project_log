import numpy as np

import cv2

from math import sqrt

_LINE_THICKNESS_SCALING = 500.0

np.random.seed(0)
RAND_COLORS = np.random.randint(50, 255, (64, 3), "int")  # used for class visu
RAND_COLORS[0] = [220, 220, 220]



_TEXT_THICKNESS_SCALING = 700.0
_TEXT_SCALING = 520.0



