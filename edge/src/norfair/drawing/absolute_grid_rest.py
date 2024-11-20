from functools import lru_cache

import numpy as np

from norfair.camera_motion import CoordinatesTransformation

from .color import Color, ColorType
from .drawer import Drawer


@lru_cache(maxsize=4)

