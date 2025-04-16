# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

from pathlib import Path

import numpy as np
import torch

from tqdm import tqdm

from deeplite_torch_zoo.src.object_detection.eval.utils import (
    box_iou,
    non_max_suppression,
    ap_per_class,
)
from deeplite_torch_zoo.src.object_detection.datasets.utils import xywh2xyxy
from deeplite_torch_zoo.src.object_detection.eval.v8.v8_nms import (
    non_max_suppression as non_max_suppression_v8,
)
from deeplite_torch_zoo.utils import LOGGER, smart_inference_mode, Profile, TQDM_BAR_FORMAT








@smart_inference_mode()