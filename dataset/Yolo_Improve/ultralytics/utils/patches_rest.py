# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Monkey patches to update/extend functionality of existing functions."""

import time
from pathlib import Path

import cv2
import numpy as np
import torch

# OpenCV Multilanguage-friendly functions ------------------------------------------------------------------------------
_imshow = cv2.imshow  # copy to avoid recursion errors








# PyTorch functions ----------------------------------------------------------------------------------------------------
_torch_save = torch.save  # copy to avoid recursion errors

