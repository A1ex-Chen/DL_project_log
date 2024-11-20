"""
YOLOX-specific modules
Source: https://github.com/iscyy/yoloair
"""

import math

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from deeplite_torch_zoo.src.object_detection.yolo.common import *
from deeplite_torch_zoo.src.object_detection.yolo.experimental import *
from deeplite_torch_zoo.src.object_detection.yolo.losses.yolox.yolox_loss import *
from deeplite_torch_zoo.utils import LOGGER


class IOUloss(nn.Module):



class DetectX(nn.Module):
    stride = [8, 16, 32]
    onnx_dynamic = False  # ONNX export parameter
    export = False










    @staticmethod

    @torch.no_grad()

    @staticmethod

    @staticmethod

    @staticmethod