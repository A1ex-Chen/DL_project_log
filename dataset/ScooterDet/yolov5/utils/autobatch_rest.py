# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Auto-batch utils
"""

from copy import deepcopy

import numpy as np
import torch

from utils.general import LOGGER, colorstr
from utils.torch_utils import profile



