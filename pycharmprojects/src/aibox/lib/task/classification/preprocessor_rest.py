from typing import Tuple, Dict, Any, Union

import PIL
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import transforms

from ...preprocessor import Preprocessor as Base


class Preprocessor(Base):

    PROCESS_KEY_EVAL_CENTER_CROP_RATIO = 'eval_center_crop_ratio'




    @classmethod

    @classmethod