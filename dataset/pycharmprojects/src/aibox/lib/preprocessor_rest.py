from math import ceil
from typing import Tuple, Dict, Any, Union

import PIL
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_tensor


class Preprocessor:

    PROCESS_KEY_IS_TRAIN_OR_EVAL = 'is_train_or_eval'
    PROCESS_KEY_ORIGIN_WIDTH = 'origin_width'
    PROCESS_KEY_ORIGIN_HEIGHT = 'origin_height'
    PROCESS_KEY_WIDTH_SCALE = 'width_scale'
    PROCESS_KEY_HEIGHT_SCALE = 'height_scale'
    PROCESS_KEY_RIGHT_PAD = 'right_pad'
    PROCESS_KEY_BOTTOM_PAD = 'bottom_pad'




    @staticmethod

    @classmethod

    @classmethod

    @classmethod